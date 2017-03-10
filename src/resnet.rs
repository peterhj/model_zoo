use arraydiff::prelude::*;
use arraydiff::ops::*;
use arraydiff::ops::cuda::*;
use async_execution::*;
use densearray::prelude::*;
use devicemem_cuda::prelude::*;

use std::rc::{Rc};

const EPSILON: f64 = 1.0e-12;

pub fn linear_op_gpu<Op>(param_vars: &mut VarSet, x_: Rc<Op>) -> () where Op: ArrayOp<DeviceBatchArray1d<f32>> {
  unimplemented!();
}

pub fn conv2d_op_gpu<Op>(shape: ConvShape<(usize, usize)>, bias: bool, param_vars: &mut VarSet, x_: Rc<Op>) -> () /*Rc<impl ArrayOp<DeviceBatchArray3d<f32>>>*/ where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  unimplemented!();
}

pub fn batch_norm_conv2d_op_gpu<Op>(shape: ConvShape<(usize, usize)>, stats_cfg: BatchStatsConfig, stats_ctrl: &mut BatchStatsControl, param_vars: &mut VarSet, x_: Rc<Op>) -> Rc<impl ArrayOp<DeviceBatchArray3d<f32>>> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  let w1 = src({
    let x = x_.data();
    move |txn, node| {
      let x_dim = x.val.get(txn, node).dim();
      match shape.axes {
        Axes((0, 1)) => DeviceArray4d::<f32>::zeros(shape.conv2d_kernel_dim(x_dim), DeviceStream::implicit().conn()),
        _ => unimplemented!(),
      }
    }
  }).initialize(init_val(kaiming_conv2d_init_gpu(shape.axes)));
  param_vars.insert_all(&w1.vars());
  let a1 = src({
    let x = x_.data();
    move |txn, node| {
      let x_dim = x.val.get(txn, node).dim();
      match shape.axes {
        Axes((0, 1)) => DeviceArray1d::<f32>::zeros(shape.conv2d_output_dim(x_dim).2, DeviceStream::implicit().conn()),
        _ => unimplemented!(),
      }
    }
  }).initialize(init_val(|_, w: &mut DeviceArray1d<f32>| w.as_view_mut().set_constant(1.0, DeviceStream::implicit().conn())));
  param_vars.insert_all(&a1.vars());
  let b1 = src({
    let x = x_.data();
    move |txn, node| {
      let x_dim = x.val.get(txn, node).dim();
      match shape.axes {
        Axes((0, 1)) => DeviceArray1d::<f32>::zeros(shape.conv2d_output_dim(x_dim).2, DeviceStream::implicit().conn()),
        _ => unimplemented!(),
      }
    }
  }).initialize(init_val(|_, w: &mut DeviceArray1d<f32>| w.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())));
  param_vars.insert_all(&b1.vars());

  let y = w1.conv(shape, x_);
  let y_stats = batch_stats(shape.axes, stats_cfg, stats_ctrl, y.clone());
  let y = y.elem_normalize(shape.axes, EPSILON, y_stats.mean.clone(), y_stats.var.clone());
  // FIXME: this should really be a "broadcast mult-add" op.
  let y = a1.elem_mult_add(/*Axes((0, 1)),*/ y, b1);
  y
}

pub fn residual_conv2d_op_gpu<Op>(axes: Axes<(usize, usize)>, stats_cfg: BatchStatsConfig, stats_ctrl: &mut BatchStatsControl, param_vars: &mut VarSet, x_: Rc<Op>) -> Rc<impl ArrayOp<DeviceBatchArray3d<f32>>> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  let shape1 = ConvShape{
    axes:       axes,
    kernel:     (3, 3),
    stride:     (1, 1),
    zero_pad:   true,
    filters:    None,
  };
  let shape2 = ConvShape{
    axes:       axes,
    kernel:     (3, 3),
    stride:     (1, 1),
    zero_pad:   true,
    filters:    None,
  };
  let y_ = batch_norm_conv2d_op_gpu(shape1, stats_cfg, stats_ctrl, param_vars, x_.clone());
  // TODO: add rectifier.
  //let y_ = y_.rect();
  let y_ = batch_norm_conv2d_op_gpu(shape2, stats_cfg, stats_ctrl, param_vars, y_);
  let y_ = x_.add(y_);
  // TODO: add rectifier.
  //let y_ = y_.rect();
  y_
}

pub fn proj_residual_conv2d_op_gpu<Op>(axes: Axes<(usize, usize)>, stride: (usize, usize), filter: usize, stats_cfg: BatchStatsConfig, stats_ctrl: &mut BatchStatsControl, param_vars: &mut VarSet, x_: Rc<Op>) -> Rc<impl ArrayOp<DeviceBatchArray3d<f32>>> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  let proj_shape = ConvShape{
    axes:       axes,
    kernel:     (1, 1),
    stride:     stride,
    zero_pad:   true,
    filters:    Some(filter),
  };
  let shape1 = ConvShape{
    axes:       axes,
    kernel:     (3, 3),
    stride:     stride,
    zero_pad:   true,
    filters:    Some(filter),
  };
  let shape2 = ConvShape{
    axes:       axes,
    kernel:     (3, 3),
    stride:     (1, 1),
    zero_pad:   true,
    filters:    None,
  };
  let proj_x_ = batch_norm_conv2d_op_gpu(proj_shape, stats_cfg, stats_ctrl, param_vars, x_.clone());
  let y_ = batch_norm_conv2d_op_gpu(shape1, stats_cfg, stats_ctrl, param_vars, x_.clone());
  // TODO: add rectifier.
  //let y_ = y_.rect();
  let y_ = batch_norm_conv2d_op_gpu(shape2, stats_cfg, stats_ctrl, param_vars, y_);
  let y_ = proj_x_.add(y_);
  // TODO: add rectifier.
  //let y_ = y_.rect();
  y_
}

pub fn max_pool_op_gpu() -> () {
  unimplemented!();
}

pub fn avg_pool_op_gpu() -> () {
  unimplemented!();
}

pub fn cifar10_resnet20_loss_gpu(batch_sz: usize) -> () {
  let mut input_vars = var_set();
  let mut label_vars = var_set();
  let mut prob_vars = var_set();
  let mut loss_vars = var_set();

  let mut const_vars = var_set();
  let mut param_vars = var_set();

  let frame_dim = (32, 32, 3);
  let frame_len = frame_dim.flat_len();
  let stats_cfg = BatchStatsConfig{average: BatchStatsAverage::Geometric(0.01)};
  let mut stats_ctrl = BatchStatsControl::new();

  let input = src(move |_, _| DeviceBatchIoMem::<u8>::with_capacity(frame_len, batch_sz));
  input_vars = input_vars.union(input.vars());
  let x = input.reify(frame_dim).cast();
  let x_scale = src(move |_, _| 1.0_f32 / 255.0)
    .initialize(init_val(|_, c: &mut f32| {
      *c = 1.0 / 255.0;
    }));
  const_vars = const_vars.union(x_scale.vars());
  let x = x_scale.elem_mult(x);

  // TODO

  unimplemented!();
}

pub fn imagenet_resnet18_loss_gpu(batch_sz: usize) -> () {
  unimplemented!();
}
