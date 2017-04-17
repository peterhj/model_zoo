use arraydiff::prelude::*;
use arraydiff::ops::*;
use arraydiff::ops::cuda::*;
use async_execution::*;
use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use superlearn::templates::*;

use std::rc::{Rc};

const EPSILON: f64 = 1.0e-6;

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

pub fn batch_norm_conv2d_op_gpu<Op>(x_dim: (usize, usize, usize), shape: ConvShape<(usize, usize)>, stats_cfg: BatchStatsConfig, stats_ctrl: &mut BatchStatsControl, param_vars: &mut VarSet, const_vars: &mut VarSet, x_: Rc<Op>) -> Rc<impl ArrayOp<DeviceBatchArray3d<f32>>> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
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
  // NOTE: The batch stats accumulators are unreachable from the sink,
  // so they are not really considered "const".
  const_vars.insert_all(&y_stats.mean_fixed.vars());
  const_vars.insert_all(&y_stats.var_fixed.vars());
  let y_ = y_.elem_normalize(shape.axes, EPSILON, y_stats.mean_branch.clone(), y_stats.var_branch.clone());
  //let y_ = y_.elem_normalize(shape.axes, EPSILON, y_stats.mean.clone(), y_stats.var.clone());
  // TODO: this should really be a "broadcast mult-add" op.
  let y_ = a1.elem_mult_add(/*Axes((0, 1)),*/ y_, b1);
  y_
}
