from collections import defaultdict
import csv
import os
from termcolor import colored
import yaml

import wandb

COMMON_TRAIN_FORMAT = [('episode', 'E', 'int'), ('step', 'S', 'int'),
                       ('training_return', 'R', 'float'),
                       ('fps', 'F', 'float'), ('duration', 'D', 'time')]

COMMON_EVAL_FORMAT = [('episode', 'E', 'int'), ('step', 'S', 'int'),
                      ('average_test_return', 'R', 'float'),
                      ('fps', 'FPS', 'float')]

AGENT_TRAIN_FORMAT = {
    'opirl': [('batch_rewards', 'BR', 'float'),
              ('discriminator_loss', 'DLOSS', 'float'),
              ('discriminator_accuracy', 'DACC', 'float')],
    'sac': [('batch_reward', 'BR', 'float'), ('actor_loss', 'ALOSS', 'float'),
            ('critic_Q_loss', 'CLOSS', 'float'),
            ('alpha_loss', 'TLOSS', 'float'), ('ent', 'ENT', 'float')],
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self, keep_prefix=False):
        data = dict()
        for key, meter in self._meters.items():
            if not keep_prefix:
                if key.startswith('train'):
                    key = key[len('train') + 1:]
                elif key.startswith('common'):
                    key = key[len('common') + 1:]
                else:
                    key = key[len('eval') + 1:]
                key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value:6d}'
        elif ty == 'float':
            return f'{key}: {value:8.3f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        if prefix == 'train':
            return
        elif prefix == 'common':
            prefix = colored(prefix, 'yellow')
        elif prefix == 'eval':
            prefix = colored(prefix, 'green')
        else:
            raise NotImplementedError

        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self,
             step,
             prefix,
             save=True,
             dump_to_csv=True,
             dump_to_console=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            if dump_to_csv:
                self._dump_to_csv(data)
            if dump_to_console:
                self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self,
                 log_dir,
                 args,
                 log_frequency=1e4,
                 log_video=False,
                 agent='sac'):
        self._agent = agent

        agents = agent.split('_')
        if len(agents) > 1 and agent != 'value_dice':
            agent = agents[0]

        if not hasattr(args, 'wandb_entity'):
            args = Logger.get_argument()
            args = args.parse_args([])
        self._set_from_args(args)

        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'),
                                     formating=AGENT_TRAIN_FORMAT[agent])
        self._common_mg = MetersGroup(os.path.join(log_dir, 'common'),
                                      formating=COMMON_TRAIN_FORMAT)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'),
                                    formating=COMMON_EVAL_FORMAT)

        self._log_dir = log_dir
        self._log_frequency = log_frequency
        self._log_video = log_video

        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f)
        # args.append_flags_into_file(os.path.join(log_dir, 'config.txt'))
        if self._log_to_wandb:
            wandb.init(entity=self._entity,
                       project=self._project,
                       name=self._name,
                       config=args)

    def _set_from_args(self, args):
        # experiment settings
        self._entity = args.wandb_entity
        self._project = args.wandb_project
        if not hasattr(args, 'env_name'):
            self._name = f'gr_{self._agent}'
        else:
            self._name = f'gr_{self._agent}_{args.env_name.split("-")[0].lower()}'
        self._log_to_wandb = args.log_to_wandb

    def _assert_key(self, key):
        assert key.startswith(('train', 'eval', 'common'))

    def _should_log(self, step, log_frequency=5):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _try_wandb_log(self, mgs, step):
        for key, mg in mgs.items():
            data = mg._prime_meters(keep_prefix=True)
            data[f'{key}/step'] = step
            wandb.log(data, commit=True)

    def _try_wandb_video(self, key, frames, step, fps=30, fmt='gif'):
        wandb.log({'video': wandb.Video(frames, fps=fps, format=fmt)})

    def _get_mg(self, key):
        if key.startswith('train'):
            return self._train_mg
        elif key.startswith('common'):
            return self._common_mg
        elif key.startswith('eval'):
            return self._eval_mg
        else:
            raise NotImplementedError

    def log(self, key, value, n=1):
        self._assert_key(key)
        mg = self._get_mg(key)
        mg.log(key, value, n)

    def log_video(self, key, frames, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self._assert_key(key)
        if self._log_to_wandb:
            self._try_wandb_video(key, frames, step)

    def log_image(self, key, path):
        if self._log_to_wandb:
            wandb.log({key: wandb.Image(path)})

    def dump(self,
             step,
             save=True,
             dump_eval=False,
             dump_to_csv=False,
             dump_to_console=True,
             flush_tb=False,
             flush_sw=False,
             log_frequency=1):
        self._dump_wandb(step,
                         save=save,
                         dump_eval=dump_eval,
                         dump_to_csv=dump_to_csv,
                         dump_to_console=dump_to_console,
                         log_frequency=log_frequency)

    def _dump_wandb(self,
                    step,
                    save=True,
                    dump_eval=False,
                    dump_to_csv=False,
                    dump_to_console=True,
                    log_frequency=1):
        common_mg = self._get_mg('common')
        train_mg = self._get_mg('train')
        mgs = {'common': common_mg, 'train': train_mg}
        if dump_eval:
            eval_mg = self._get_mg('eval')
            mgs['eval'] = eval_mg

        if self._log_to_wandb and step % log_frequency == 0:
            self._try_wandb_log(mgs, step)

        common_mg.dump(step,
                       'common',
                       save,
                       dump_to_csv=dump_to_csv,
                       dump_to_console=dump_to_console)
        train_mg.dump(step, 'train', save, dump_to_console=dump_to_console)
        if dump_eval:
            eval_mg.dump(step,
                         'eval',
                         save,
                         dump_to_csv=dump_to_csv,
                         dump_to_console=dump_to_console)
