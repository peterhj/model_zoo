use arraydiff::prelude::*;
use arraydiff::ops::*;
use arraydiff::ops::cuda::*;
use async_execution::*;
use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use superlearn::templates::*;

use std::rc::{Rc};

const EPSILON: f64 = 1.0e-12;

pub fn linear_op_gpu<Op>(in_dim: usize, filters: usize, bias: bool, param_vars: &mut VarSet, x_: Rc<Op>) -> Rc<impl ArrayOp<DeviceBatchArray1d<f32>>> where Op: 'static + ArrayOp<DeviceBatchArray1d<f32>> {
  let w1 = src({
    //let x = x_.data();
    move |txn, node| {
      //let x_dim = x.val.get(txn, node).dim();
      DeviceArray2d::<f32>::zeros((filters, in_dim), DeviceStream::implicit().conn())
    }
  //}).initialize(init_val(normal_linear_init_gpu(0.0, 0.01)));
  }).initialize(init_val(xavier_linear_init_gpu()));
  param_vars.insert_all(&w1.vars());

  if !bias {
    let y_ = w1.mult(x_);
    y_
  } else {
    unimplemented!();
  }
}

pub fn conv2d_op_gpu<Op>(x_dim: (usize, usize, usize), shape: ConvShape<(usize, usize)>, bias: bool, param_vars: &mut VarSet, x_: Rc<Op>) -> Rc<impl ArrayOp<DeviceBatchArray3d<f32>>> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  let w1 = src({
    //let x = x_.data();
    move |txn, node| {
      //let x_dim = x.val.get(txn, node).dim();
      match shape.axes {
        Axes((0, 1)) => DeviceArray4d::<f32>::zeros(shape.conv2d_kernel_dim(x_dim), DeviceStream::implicit().conn()),
        _ => unimplemented!(),
      }
    }
  }).initialize(init_val(kaiming_conv2d_init_gpu(shape.axes)));
  param_vars.insert_all(&w1.vars());

  if !bias {
    let y_ = w1.conv(shape, x_);
    y_
  } else {
    unimplemented!();
  }
}

pub fn batch_norm_conv2d_op_gpu<Op>(x_dim: (usize, usize, usize), shape: ConvShape<(usize, usize)>, stats_cfg: BatchStatsConfig, stats_ctrl: &mut BatchStatsControl, param_vars: &mut VarSet, x_: Rc<Op>) -> Rc<impl ArrayOp<DeviceBatchArray3d<f32>>> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  let w1 = src({
    //let x = x_.data();
    move |txn, node| {
      //let x_dim = x.val.get(txn, node).dim();
      match shape.axes {
        Axes((0, 1)) => DeviceArray4d::<f32>::zeros(shape.conv2d_kernel_dim(x_dim), DeviceStream::implicit().conn()),
        _ => unimplemented!(),
      }
    }
  }).initialize(init_val(kaiming_conv2d_init_gpu(shape.axes)));
  param_vars.insert_all(&w1.vars());
  let a1 = src({
    //let x = x_.data();
    move |txn, node| {
      //let x_dim = x.val.get(txn, node).dim();
      match shape.axes {
        Axes((0, 1)) => DeviceArray1d::<f32>::zeros(shape.conv2d_output_dim(x_dim).2, DeviceStream::implicit().conn()),
        _ => unimplemented!(),
      }
    }
  }).initialize(init_val(|_, w: &mut DeviceArray1d<f32>| w.as_view_mut().set_constant(1.0, DeviceStream::implicit().conn())));
  param_vars.insert_all(&a1.vars());
  let b1 = src({
    //let x = x_.data();
    move |txn, node| {
      //let x_dim = x.val.get(txn, node).dim();
      match shape.axes {
        Axes((0, 1)) => DeviceArray1d::<f32>::zeros(shape.conv2d_output_dim(x_dim).2, DeviceStream::implicit().conn()),
        _ => unimplemented!(),
      }
    }
  }).initialize(init_val(|_, w: &mut DeviceArray1d<f32>| w.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())));
  param_vars.insert_all(&b1.vars());

  let y_ = w1.conv(shape, x_);
  let y_stats = batch_stats(shape.axes, stats_cfg, stats_ctrl, y_.clone());
  let y_ = y_.elem_normalize(shape.axes, EPSILON, y_stats.mean.clone(), y_stats.var.clone());
  // TODO: this should really be a "broadcast mult-add" op.
  let y_ = a1.elem_mult_add(/*Axes((0, 1)),*/ y_, b1);
  y_
}

pub fn residual_conv2d_op_gpu<Op>(x_dim: (usize, usize, usize), axes: Axes<(usize, usize)>, stats_cfg: BatchStatsConfig, stats_ctrl: &mut BatchStatsControl, param_vars: &mut VarSet, x_: Rc<Op>) -> Rc<impl ArrayOp<DeviceBatchArray3d<f32>>> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
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
  let y_dim = shape1.conv2d_output_dim(x_dim);
  assert_eq!(y_dim, shape2.conv2d_output_dim(y_dim));
  let y_ = batch_norm_conv2d_op_gpu(x_dim, shape1, stats_cfg, stats_ctrl, param_vars, x_.clone());
  let y_ = y_.rect();
  let y_ = batch_norm_conv2d_op_gpu(x_dim, shape2, stats_cfg, stats_ctrl, param_vars, y_);
  let y_ = x_.add(y_);
  y_
}

pub fn proj_residual_conv2d_op_gpu<Op>(x_dim: (usize, usize, usize), axes: Axes<(usize, usize)>, stride: (usize, usize), filters: usize, stats_cfg: BatchStatsConfig, stats_ctrl: &mut BatchStatsControl, param_vars: &mut VarSet, x_: Rc<Op>) -> Rc<impl ArrayOp<DeviceBatchArray3d<f32>>> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  let proj_shape = ConvShape{
    axes:       axes,
    kernel:     (1, 1),
    stride:     stride,
    zero_pad:   true,
    filters:    Some(filters),
  };
  let shape1 = ConvShape{
    axes:       axes,
    kernel:     (3, 3),
    stride:     stride,
    zero_pad:   true,
    filters:    Some(filters),
  };
  let shape2 = ConvShape{
    axes:       axes,
    kernel:     (3, 3),
    stride:     (1, 1),
    zero_pad:   true,
    filters:    None,
  };
  let y_dim = proj_shape.conv2d_output_dim(x_dim);
  assert_eq!(y_dim, shape1.conv2d_output_dim(x_dim));
  assert_eq!(y_dim, shape2.conv2d_output_dim(y_dim));
  let proj_x_ = batch_norm_conv2d_op_gpu(x_dim, proj_shape, stats_cfg, stats_ctrl, param_vars, x_.clone());
  let y_ = batch_norm_conv2d_op_gpu(x_dim, shape1, stats_cfg, stats_ctrl, param_vars, x_.clone());
  let y_ = y_.rect();
  let y_ = batch_norm_conv2d_op_gpu(y_dim, shape2, stats_cfg, stats_ctrl, param_vars, y_);
  let y_ = proj_x_.add(y_);
  y_
}

pub fn max_pool_op_gpu() -> () {
  unimplemented!();
}

pub fn avg_pool_op_gpu() -> () {
  unimplemented!();
}

pub fn cifar10_resnet20_model_gpu(batch_sz: usize) -> (CategoricalNLLLoss<DeviceBatchIoMem<u8>, DeviceIoBatch<u32>, DeviceBatchArray1d<f32>, DeviceIoBatch<f32>>, BatchStatsControl) {
  let mut input_vars = var_set();
  let mut label_vars = var_set();
  let mut prob_vars = var_set();
  let mut loss_vars = var_set();

  let mut const_vars = var_set();
  let mut param_vars = var_set();

  let frame_dim = (32, 32, 3);
  let frame_len = frame_dim.flat_len();
  let conv_axes = Axes((0, 1));
  let stats_cfg = BatchStatsConfig{average: BatchStatsAverage::Geometric(0.01)};
  let mut stats_ctrl = BatchStatsControl::new();

  let input_ = src(move |_, _| DeviceBatchIoMem::<u8>::with_capacity(frame_len, batch_sz));
  input_vars = input_vars.union(input_.vars());
  let label_ = src(move |_, _| DeviceIoBatch::<u32>::zeros(batch_sz, DeviceStream::implicit().conn()));
  label_vars = label_vars.union(label_.vars());

  let x_ = input_.reify(frame_dim).cast();
  let x_scale = src(move |_, _| 0.0_f32)
    .initialize(init_val(|_, c: &mut f32| { *c = 1.0 / 255.0; }));
  const_vars = const_vars.union(x_scale.vars());
  let x_ = x_scale.elem_mult(x_);
  let y_ = x_;

  let y_ = batch_norm_conv2d_op_gpu(frame_dim, ConvShape{
    axes:       conv_axes,
    kernel:     (5, 5),
    stride:     (1, 1),
    zero_pad:   true,
    filters:    Some(16),
  }, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();

  let conv1_dim = (32, 32, 16);
  let y_ = residual_conv2d_op_gpu(conv1_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv1_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv1_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();

  let conv2_dim = (16, 16, 32);
  //let y_ = proj_residual_conv2d_op_gpu(conv1_dim, conv_axes, (2, 2), 32, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = max_pool(PoolShape{
      axes:     conv_axes,
      kernel:   (3, 3),
      stride:   (2, 2),
      zero_pad: true,
  }, y_);
  let y_ = proj_residual_conv2d_op_gpu((16, 16, 16), conv_axes, (1, 1), 32, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv2_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv2_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();

  let conv3_dim = (8, 8, 64);
  //let y_ = proj_residual_conv2d_op_gpu(conv2_dim, conv_axes, (2, 2), 64, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = max_pool(PoolShape{
      axes:     conv_axes,
      kernel:   (3, 3),
      stride:   (2, 2),
      zero_pad: true,
  }, y_);
  let y_ = proj_residual_conv2d_op_gpu((8, 8, 32), conv_axes, (1, 1), 64, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv3_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv3_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();

  let y_ = avg_pool(PoolShape{
      axes:     conv_axes,
      kernel:   (8, 8),
      stride:   (8, 8),
      zero_pad: false,
  }, y_);

  let y_ = y_.flatten();
  //let y_ = linear_op_gpu(conv3_dim.flat_len(), 10, false, &mut param_vars, y_);
  let y_ = linear_op_gpu(64, 10, false, &mut param_vars, y_);

  let (prob_, loss_) = softmax_nll_loss(y_, label_.clone());
  prob_vars = prob_vars.union(prob_.vars());
  loss_vars = loss_vars.union(loss_.vars());
  let scalar_loss_ = batch_sum(loss_.clone());
  let obj = sink(scalar_loss_);

  let loss = CategoricalNLLLoss{
    obj:          obj,
    input:        input_,
    label:        label_,
    prob:         prob_,
    loss:         loss_,
    input_vars:   input_vars,
    label_vars:   label_vars,
    prob_vars:    prob_vars,
    loss_vars:    loss_vars,
    const_vars:   const_vars,
    param_vars:   param_vars,
  };
  (loss, stats_ctrl)
}

pub fn imagenet_resnet18_model_gpu(batch_sz: usize) -> (CategoricalNLLLoss<DeviceBatchIoMem<u8>, DeviceIoBatch<u32>, DeviceBatchArray1d<f32>, DeviceIoBatch<f32>>, BatchStatsControl) {
  let mut input_vars = var_set();
  let mut label_vars = var_set();
  let mut prob_vars = var_set();
  let mut loss_vars = var_set();

  let mut const_vars = var_set();
  let mut param_vars = var_set();

  let frame_dim = (224, 224, 3);
  let frame_len = frame_dim.flat_len();
  let conv_axes = Axes((0, 1));
  let stats_cfg = BatchStatsConfig{average: BatchStatsAverage::Geometric(0.01)};
  let mut stats_ctrl = BatchStatsControl::new();

  let input_ = src(move |_, _| DeviceBatchIoMem::<u8>::with_capacity(frame_len, batch_sz));
  input_vars = input_vars.union(input_.vars());
  let label_ = src(move |_, _| DeviceIoBatch::<u32>::zeros(batch_sz, DeviceStream::implicit().conn()));
  label_vars = label_vars.union(label_.vars());

  let x_ = input_.reify(frame_dim).cast();
  let x_scale = src(move |_, _| 0.0_f32)
    .initialize(init_val(|_, c: &mut f32| { *c = 1.0 / 255.0; }));
  const_vars = const_vars.union(x_scale.vars());
  let x_ = x_scale.elem_mult(x_);
  let y_ = x_;

  let y_ = batch_norm_conv2d_op_gpu(frame_dim, ConvShape{
    axes:       conv_axes,
    kernel:     (7, 7),
    stride:     (2, 2),
    zero_pad:   true,
    filters:    Some(64),
  }, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();

  let y_ = max_pool(PoolShape{
    axes:       conv_axes,
    kernel:     (3, 3),
    stride:     (2, 2),
    zero_pad:   true,
  }, y_);

  let conv1_dim = (56, 56, 64);
  let y_ = residual_conv2d_op_gpu(conv1_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv1_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();

  let conv2_dim = (28, 28, 128);
  let y_ = proj_residual_conv2d_op_gpu(conv1_dim, conv_axes, (2, 2), 128, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv2_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();

  let conv3_dim = (14, 14, 256);
  let y_ = proj_residual_conv2d_op_gpu(conv2_dim, conv_axes, (2, 2), 256, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv3_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();

  let conv4_dim = (7, 7, 512);
  let y_ = proj_residual_conv2d_op_gpu(conv3_dim, conv_axes, (2, 2), 512, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();
  let y_ = residual_conv2d_op_gpu(conv4_dim, conv_axes, stats_cfg, &mut stats_ctrl, &mut param_vars, y_).rect();

  let y_ = avg_pool(PoolShape{
    axes:       conv_axes,
    kernel:     (7, 7),
    stride:     (7, 7),
    zero_pad:   false,
  }, y_);

  let y_ = y_.flatten();
  let y_ = linear_op_gpu(512, 1000, false, &mut param_vars, y_);

  let (prob_, loss_) = softmax_nll_loss(y_, label_.clone());
  prob_vars = prob_vars.union(prob_.vars());
  loss_vars = loss_vars.union(loss_.vars());
  let scalar_loss_ = batch_sum(loss_.clone());
  let obj = sink(scalar_loss_);

  let loss = CategoricalNLLLoss{
    obj:          obj,
    input:        input_,
    label:        label_,
    prob:         prob_,
    loss:         loss_,
    input_vars:   input_vars,
    label_vars:   label_vars,
    prob_vars:    prob_vars,
    loss_vars:    loss_vars,
    const_vars:   const_vars,
    param_vars:   param_vars,
  };
  (loss, stats_ctrl)
}
