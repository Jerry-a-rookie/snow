from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import hashlib
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import csv
from collections import defaultdict

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    @staticmethod
    def _result_fieldnames():
        return [
            'setting', 'compare_group', 'data', 'model_id', 'model', 'seed',
            'des', 'use_norm', 'individual', 'hidden_size', 'hidden_dim',
            'batch_size', 'learning_rate', 'lradj', 'dropout',
            'snow_active', 'snow_config', 'snow_clusters', 'snow_temperature',
            'snow_residual_scale', 'snow_learnable_scale',
            'train_noise_type', 'train_noise_channel_ratio', 'train_noise_std',
            'train_noise_seed', 'train_noise_channel_count',
            'train_noise_channel_hash', 'channel_shuffle_ratio',
            'eval_channel_shuffle_ratio', 'channel_shuffle_seed',
            'channel_shuffle_count', 'channel_shuffle_hash',
            'seq_len', 'pred_len',
            'd_model', 'd_core', 'd_ff', 'e_layers',
            'mse', 'mae', 'rmse', 'mape', 'metric_space',
            'params', 'trainable_params', 'flops_or_macs',
            'infer_time_ms', 'gpu_memory_mb', 'epoch_time_sec'
        ]

    @staticmethod
    def _schema_path(path):
        stem, ext = os.path.splitext(path)
        return f"{stem}_schema_snownet{ext or '.csv'}"

    def _resolve_result_path(self, result_path, fieldnames):
        if not os.path.exists(result_path):
            return result_path, False
        with open(result_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header == fieldnames:
            return result_path, True

        schema_path = self._schema_path(result_path)
        schema_exists = os.path.exists(schema_path)
        if schema_exists:
            with open(schema_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                schema_header = next(reader, None)
            if schema_header != fieldnames:
                raise ValueError(
                    f"Existing result file has incompatible header: {schema_path}. "
                    "Please move or rename it before writing new Snow comparison results."
                )
        print(
            f"Result header mismatch in {result_path}; writing Snow comparison rows to {schema_path}"
        )
        return schema_path, schema_exists

    def _snow_result_config(self):
        snow_active = 'Snow' in self.args.model
        if snow_active:
            snow_config = (
                f"d_core={self.args.d_core},clusters={self.args.snow_clusters},"
                f"temperature={self.args.snow_temperature},"
                f"residual_scale={self.args.snow_residual_scale},"
                f"learnable_scale={self.args.snow_learnable_scale}"
            )
        else:
            snow_config = ''
        return int(snow_active), snow_config

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _checkpoint_dir(self, setting):
        """Return a Windows-safe checkpoint directory for one experiment."""
        path = os.path.join(self.args.checkpoints, setting)
        checkpoint_file = os.path.abspath(os.path.join(path, 'checkpoint.pth'))
        if os.name == 'nt' and len(checkpoint_file) >= 240:
            digest = hashlib.sha1(setting.encode('utf-8')).hexdigest()[:16]
            path = os.path.join(self.args.checkpoints, f'setting_{digest}')
            print(
                'Checkpoint path is too long for Windows; using short directory: '
                f'{path}'
            )
        return path

    @staticmethod
    def _is_highdim_traffic(data_set):
        return bool(getattr(data_set, 'is_highdim_traffic', False))

    @staticmethod
    def _masked_mae(predictions, targets):
        mask = targets.ne(0) & torch.isfinite(targets)
        if not torch.any(mask):
            return predictions.sum() * 0.0
        return torch.abs(predictions[mask] - targets[mask]).mean()

    def _forecast_loss(self, outputs, targets, data_set, criterion):
        if self._is_highdim_traffic(data_set):
            outputs = data_set.inverse_transform(outputs)
            return self._masked_mae(outputs, targets)
        return criterion(outputs, targets)

    def _prepare_forecast_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if 'PEMS' in self.args.data or 'Solar' in self.args.data:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    @staticmethod
    def _count_profile_macs(model, forward_fn):
        macs = defaultdict(int)
        handles = []

        def linear_hook(module, inputs, output):
            if torch.is_tensor(output):
                macs['total'] += output.numel() * module.in_features

        def conv_hook(module, inputs, output):
            if not torch.is_tensor(output):
                return
            kernel_ops = module.in_channels // module.groups
            for kernel_size in module.kernel_size:
                kernel_ops *= kernel_size
            macs['total'] += output.numel() * kernel_ops

        def rnn_hook(module, inputs, output):
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            if module.batch_first:
                batch_size, seq_len = x.shape[0], x.shape[1]
            else:
                seq_len, batch_size = x.shape[0], x.shape[1]
            gate_multiplier = 4 if isinstance(module, nn.LSTM) else 3 if isinstance(module, nn.GRU) else 1
            directions = 2 if module.bidirectional else 1
            total = 0
            for layer in range(module.num_layers):
                layer_input_size = module.input_size if layer == 0 else module.hidden_size * directions
                total += (
                    batch_size
                    * seq_len
                    * directions
                    * gate_multiplier
                    * module.hidden_size
                    * (layer_input_size + module.hidden_size)
                )
            macs['total'] += total

        for module in model.modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                handles.append(module.register_forward_hook(rnn_hook))

        try:
            with torch.no_grad():
                forward_fn()
        finally:
            for handle in handles:
                handle.remove()
        return int(macs['total'])

    def _profile_efficiency(self, test_loader):
        profile = {
            'params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'flops_or_macs': '',
            'infer_time_ms': '',
            'gpu_memory_mb': '',
            'epoch_time_sec': getattr(self, 'epoch_time_sec', ''),
        }

        profile_batches = int(getattr(self.args, 'profile_batches', 1))
        if profile_batches <= 0:
            return profile

        try:
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
        except StopIteration:
            return profile

        batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._prepare_forecast_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark
        )

        self.model.eval()

        def forward_fn():
            return self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        try:
            profile['flops_or_macs'] = self._count_profile_macs(self.model, forward_fn)
        except Exception:
            profile['flops_or_macs'] = ''

        warmup = int(getattr(self.args, 'profile_warmup', 1))
        times = []
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            for i in range(warmup + profile_batches):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                start = time.time()
                forward_fn()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                elapsed = time.time() - start
                if i >= warmup:
                    times.append(elapsed)

        if times:
            profile['infer_time_ms'] = float(np.mean(times) * 1000.0)
        if self.device.type == 'cuda':
            profile['gpu_memory_mb'] = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
        return profile

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self._is_highdim_traffic(vali_data):
                    outputs = vali_data.inverse_transform(outputs)
                    mask = batch_y.ne(0) & torch.isfinite(batch_y)
                    total_loss.sum += torch.abs(outputs[mask] - batch_y[mask]).sum().item()
                    total_loss.count += int(mask.sum().item())
                else:
                    loss = criterion(outputs, batch_y)
                    total_loss.update(loss.item(), batch_x.size(0))
        if self._is_highdim_traffic(vali_data):
            if total_loss.count == 0:
                raise ValueError('HighDimTraffic validation split contains no nonzero finite labels')
            total_loss = total_loss.sum / total_loss.count
        else:
            total_loss = total_loss.avg
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = self._checkpoint_dir(setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_times = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self._forecast_loss(outputs, batch_y, train_data, criterion)

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self._forecast_loss(outputs, batch_y, train_data, criterion)

                loss_float = loss.item()
                train_loss.append(loss_float)
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_cost = time.time() - epoch_time
            epoch_times.append(epoch_cost)
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_cost))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.epoch_time_sec = float(np.mean(epoch_times)) if epoch_times else ''
        self.total_train_time_sec = float(np.sum(epoch_times)) if epoch_times else ''
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
        return self.model

    def test(self, setting, test=0, checkpoint_setting=None):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            checkpoint_setting = checkpoint_setting or setting
            checkpoint_path = os.path.join(
                self._checkpoint_dir(checkpoint_setting),
                'checkpoint.pth',
            )
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )

        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        mse = AverageMeter()
        mae = AverageMeter()
        traffic_sums = {
            'count': 0,
            'abs_error': 0.0,
            'squared_error': 0.0,
            'absolute_percentage_error': 0.0,
        }
        profile = self._profile_efficiency(test_loader)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self._is_highdim_traffic(test_data):
                    outputs = test_data.inverse_transform(outputs)
                    mask = batch_y.ne(0) & torch.isfinite(batch_y)
                    errors = outputs[mask] - batch_y[mask]
                    valid_targets = batch_y[mask]
                    traffic_sums['count'] += int(mask.sum().item())
                    traffic_sums['abs_error'] += torch.abs(errors).sum().item()
                    traffic_sums['squared_error'] += torch.square(errors).sum().item()
                    traffic_sums['absolute_percentage_error'] += (
                        torch.abs(errors / valid_targets).sum().item()
                    )
                else:
                    mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                    mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))

        if self._is_highdim_traffic(test_data):
            count = traffic_sums['count']
            if count == 0:
                mse = mae = rmse = mape = float('nan')
            else:
                mse = traffic_sums['squared_error'] / count
                mae = traffic_sums['abs_error'] / count
                rmse = float(np.sqrt(mse))
                mape = traffic_sums['absolute_percentage_error'] / count
            metric_space = 'original_masked_nonzero'
        else:
            mse = mse.avg
            mae = mae.avg
            rmse = float(np.sqrt(mse))
            mape = ''
            metric_space = 'model_output'
        print('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))

        result_path = getattr(self.args, 'result_path', None)
        if result_path:
            result_dir = os.path.dirname(result_path)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
            fieldnames = self._result_fieldnames()
            result_path, file_exists = self._resolve_result_path(result_path, fieldnames)
            result_data = getattr(self.args, 'result_data', '') or self.args.data
            snow_active, snow_config = self._snow_result_config()
            with open(result_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'setting': setting,
                    'compare_group': '{}_{}_sl{}_pl{}_dm{}_el{}'.format(
                        result_data,
                        self.args.model_id,
                        self.args.seq_len,
                        self.args.pred_len,
                        self.args.d_model,
                        self.args.e_layers,
                    ),
                    'data': result_data,
                    'model_id': self.args.model_id,
                    'model': self.args.model,
                    'seed': getattr(self.args, 'seed', 2021),
                    'des': getattr(self.args, 'des', ''),
                    'use_norm': getattr(self.args, 'use_norm', ''),
                    'individual': getattr(self.args, 'individual', ''),
                    'hidden_size': getattr(self.args, 'hidden_size', ''),
                    'hidden_dim': getattr(self.args, 'hidden_dim', ''),
                    'batch_size': getattr(self.args, 'batch_size', ''),
                    'learning_rate': getattr(self.args, 'learning_rate', ''),
                    'lradj': getattr(self.args, 'lradj', ''),
                    'dropout': getattr(self.args, 'dropout', ''),
                    'snow_active': snow_active,
                    'snow_config': snow_config,
                    'snow_clusters': getattr(self.args, 'snow_clusters', ''),
                    'snow_temperature': getattr(self.args, 'snow_temperature', ''),
                    'snow_residual_scale': getattr(self.args, 'snow_residual_scale', ''),
                    'snow_learnable_scale': getattr(self.args, 'snow_learnable_scale', ''),
                    'train_noise_type': getattr(self.args, 'train_noise_type', 'none'),
                    'train_noise_channel_ratio': getattr(
                        self.args, 'train_noise_channel_ratio', 0.0
                    ),
                    'train_noise_std': getattr(self.args, 'train_noise_std', ''),
                    'train_noise_seed': getattr(self.args, 'train_noise_seed', ''),
                    'train_noise_channel_count': getattr(
                        self.args, 'train_noise_channel_count', ''
                    ),
                    'train_noise_channel_hash': getattr(
                        self.args, 'train_noise_channel_hash', ''
                    ),
                    'channel_shuffle_ratio': getattr(
                        self.args, 'channel_shuffle_ratio', 0.0
                    ),
                    'eval_channel_shuffle_ratio': getattr(
                        self.args, 'active_channel_shuffle_ratio', 0.0
                    ),
                    'channel_shuffle_seed': getattr(
                        self.args, 'active_channel_shuffle_seed', ''
                    ),
                    'channel_shuffle_count': getattr(
                        self.args, 'active_channel_shuffle_count', 0
                    ),
                    'channel_shuffle_hash': getattr(
                        self.args, 'active_channel_shuffle_hash', ''
                    ),
                    'seq_len': self.args.seq_len,
                    'pred_len': self.args.pred_len,
                    'd_model': self.args.d_model,
                    'd_core': self.args.d_core,
                    'd_ff': self.args.d_ff,
                    'e_layers': self.args.e_layers,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'metric_space': metric_space,
                    'params': profile.get('params', ''),
                    'trainable_params': profile.get('trainable_params', ''),
                    'flops_or_macs': profile.get('flops_or_macs', ''),
                    'infer_time_ms': profile.get('infer_time_ms', ''),
                    'gpu_memory_mb': profile.get('gpu_memory_mb', ''),
                    'epoch_time_sec': profile.get('epoch_time_sec', ''),
                })

        return
