// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu2{emitGPUKernelLaunchBounds=true minCtasTarget=4})" | FileCheck %s

// ptxas budgets registers for the resident-block count it assumes; without
// a hint, a 256-thread kernel may be compiled at a register count that only
// fits 3 blocks per SM while the equivalent clang kernel fits 4. The target
// asks for the resident count, bounded so it stays satisfiable: a 512-thread
// block can host at most 2 such blocks worth of threads, and a kernel with
// 26KB of static shared memory at most 3 within a 96KB budget.

module attributes {gpu.container_module} {
  gpu.module @mod [#nvvm.target] {
    gpu.func @plain() kernel {
      gpu.return
    }
    gpu.func @wide() kernel {
      gpu.return
    }
    gpu.func @shared() workgroup(%buf: memref<3328xf64, 3>) kernel {
      gpu.return
    }
  }
  func.func @launch(%n: index) {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    gpu.launch_func @mod::@plain blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)
    gpu.launch_func @mod::@wide blocks in (%c1, %c1, %c1) threads in (%c512, %c1, %c1)
    gpu.launch_func @mod::@shared blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)
    return
  }
}

// CHECK: gpu.func @plain() kernel attributes {nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 4 : i32
// CHECK: gpu.func @wide() kernel attributes {nvvm.maxntid = array<i32: 512, 1, 1>, nvvm.minctasm = 2 : i32
// CHECK: gpu.func @shared() workgroup(%{{.+}}: memref<3328xf64, 3>) kernel attributes {nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 3 : i32
