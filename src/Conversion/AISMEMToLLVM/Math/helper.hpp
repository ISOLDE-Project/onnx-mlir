/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "src/Dialect/AISLLVM/AISLLVMOps.hpp"
#include "src/Dialect/AISMEM/AISMEMOps.hpp"
#include "src/Support/SpadeSupport.hpp"

#include <set>

using namespace mlir;
