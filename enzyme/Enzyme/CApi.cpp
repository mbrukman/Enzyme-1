//===- CApi.cpp - Enzyme API exported to C for external use -----------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file defines various utility functions of Enzyme for access via C
//
//===----------------------------------------------------------------------===//

#include "CApi.h"

#include "EnzymeLogic.h"

#include "llvm/Analysis/TargetLibraryInfo.h"

using namespace llvm;

TargetLibraryInfo eunwrap(LLVMTargetLibraryInfoRef P) {
  return TargetLibraryInfo(*reinterpret_cast<TargetLibraryInfoImpl *>(P));
}

TypeAnalysis &eunwrap(EnzymeTypeAnalysisRef TAR) {
  return *(TypeAnalysis *)TAR;
}
llvm::AAResults &eunwrap(EnzymeAAResultsRef AAR) {
  return *(llvm::AAResults *)AAR;
}
AugmentedReturn *eunwrap(EnzymeAugmentedReturnPtr ARP) {
  return (AugmentedReturn *)ARP;
}
EnzymeAugmentedReturnPtr ewrap(const AugmentedReturn &AR) {
  return (EnzymeAugmentedReturnPtr)(&AR);
}

ConcreteType eunwrap(CConcreteType CDT, llvm::LLVMContext &ctx) {
  switch (CDT) {
  case DT_Anything:
    return BaseType::Anything;
  case DT_Integer:
    return BaseType::Integer;
  case DT_Pointer:
    return BaseType::Pointer;
  case DT_Half:
    return ConcreteType(llvm::Type::getHalfTy(ctx));
  case DT_Float:
    return ConcreteType(llvm::Type::getHalfTy(ctx));
  case DT_Double:
    return ConcreteType(llvm::Type::getHalfTy(ctx));
  case DT_Unknown:
    return BaseType::Unknown;
  }
  llvm_unreachable("Unknown concrete type to unwrap");
}

std::vector<int> eunwrap(IntList IL) {
  std::vector<int> v;
  for (size_t i = 0; i < IL.size; i++) {
    v.push_back((int)IL.data[i]);
  }
  return v;
}
std::set<int64_t> eunwrap64(IntList IL) {
  std::set<int64_t> v;
  for (size_t i = 0; i < IL.size; i++) {
    v.insert((int64_t)IL.data[i]);
  }
  return v;
}
TypeTree eunwrap(CTypeTree CTT, llvm::LLVMContext &ctx) {
  TypeTree res;
  for (size_t i = 0; i < CTT.size; i++) {
    res.insert(eunwrap(CTT.data[i].offsets),
               eunwrap(CTT.data[i].datatype, ctx));
  }
  return res;
}

TypeTree eunwrapAndFree(CTypeTree CTT, llvm::LLVMContext &ctx) {
  TypeTree res = eunwrap(CTT, ctx);
  for (size_t i = 0; i < CTT.size; i++) {
    free(CTT.data[i].offsets.data);
  }
  free(CTT.data);
  return res;
}

CConcreteType ewrap(const ConcreteType &CT) {
  if (auto flt = CT.isFloat()) {
    if (flt->isHalfTy())
      return DT_Half;
    if (flt->isFloatTy())
      return DT_Float;
    if (flt->isDoubleTy())
      return DT_Double;
  } else {
    switch (CT.SubTypeEnum) {
    case BaseType::Integer:
      return DT_Integer;
    case BaseType::Pointer:
      return DT_Pointer;
    case BaseType::Anything:
      return DT_Anything;
    case BaseType::Unknown:
      return DT_Unknown;
    case BaseType::Float:
      llvm_unreachable("Illegal conversion of concretetype");
    }
  }
  llvm_unreachable("Illegal conversion of concretetype");
}

IntList ewrap(const std::vector<int> &offsets) {
  IntList IL;
  IL.size = offsets.size();
  IL.data = (int64_t *)malloc(IL.size * sizeof(*IL.data));
  for (size_t i = 0; i < offsets.size(); i++) {
    IL.data[i] = offsets[i];
  }
  return IL;
}

CTypeTree ewrap(const TypeTree &TT) {
  CTypeTree CTT;
  auto &mapping = TT.getMapping();
  CTT.size = mapping.size();
  CTT.data = (CDataPair *)malloc(CTT.size * sizeof(*CTT.data));
  size_t i = 0;
  for (auto &pair : mapping) {
    CTT.data[i].offsets = ewrap(pair.first);
    CTT.data[i].datatype = ewrap(pair.second);
    i++;
  }
  return CTT;
}

FnTypeInfo eunwrap(CFnTypeInfo CTI, llvm::Function *F) {
  FnTypeInfo FTI(F);
  auto &ctx = F->getContext();
  FTI.Return = eunwrap(CTI.Return, ctx);

  size_t argnum = 0;
  for (auto &arg : F->args()) {
    FTI.Arguments[&arg] = eunwrap(CTI.Arguments[argnum], ctx);
    FTI.KnownValues[&arg] = eunwrap64(CTI.KnownValues[argnum]);
    argnum++;
  }
  return FTI;
}

extern "C" {

EnzymeTypeAnalysisRef CreateTypeAnalysis(LLVMTargetLibraryInfoRef TLI,
                                         char **customRuleNames,
                                         CustomRuleType *customRules,
                                         size_t numRules) {
  TypeAnalysis *TA = new TypeAnalysis(*(new TargetLibraryInfo(eunwrap(TLI))));
  for (size_t i = 0; i < numRules; i++) {
    CustomRuleType rule = customRules[i];
    TA->CustomRules[customRuleNames[i]] =
        [=](int direction, TypeTree &returnTree,
            std::vector<TypeTree> &argTrees, CallInst *call) -> bool {
      CTypeTree creturnTree = ewrap(returnTree);
      CTypeTree *cargs = new CTypeTree[argTrees.size()];
      for (size_t i = 0; i < argTrees.size(); ++i) {
        cargs[i] = ewrap(argTrees[i]);
      }
      bool result =
          rule(direction, &creturnTree, cargs, argTrees.size(), wrap(call));
      returnTree = eunwrapAndFree(creturnTree, call->getContext());
      for (size_t i = 0; i < argTrees.size(); ++i) {
        argTrees[i] = eunwrapAndFree(cargs[i], call->getContext());
      }
      delete[] cargs;
      return result;
    };
  }
  return (EnzymeTypeAnalysisRef)TA;
}
void FreeTypeAnalysis(EnzymeTypeAnalysisRef TAR) {
  TypeAnalysis *TA = (TypeAnalysis *)TAR;
  delete &TA->TLI;
  delete TA;
}
LLVMValueRef EnzymeCreatePrimalAndGradient(
    LLVMValueRef todiff, CDIFFE_TYPE retType, CDIFFE_TYPE *constant_args,
    size_t constant_args_size, LLVMTargetLibraryInfoRef TLI,
    EnzymeTypeAnalysisRef TA, EnzymeAAResultsRef global_AA, bool returnValue,
    bool dretUsed, bool topLevel, LLVMTypeRef additionalArg,
    CFnTypeInfo typeInfo, bool *_uncacheable_args, size_t uncacheable_args_size,
    EnzymeAugmentedReturnPtr augmented, bool AtomicAdd, bool PostOpt) {
  std::vector<DIFFE_TYPE> nconstant_args((DIFFE_TYPE *)constant_args,
                                         (DIFFE_TYPE *)constant_args +
                                             constant_args_size);
  std::map<llvm::Argument *, bool> uncacheable_args;
  size_t argnum = 0;
  for (auto &arg : cast<Function>(unwrap(todiff))->args()) {
    assert(argnum < uncacheable_args_size);
    uncacheable_args[&arg] = _uncacheable_args[argnum];
    argnum++;
  }
  auto TLIpp = eunwrap(TLI);
  return wrap(CreatePrimalAndGradient(
      cast<Function>(unwrap(todiff)), (DIFFE_TYPE)retType, nconstant_args,
      TLIpp, eunwrap(TA), eunwrap(global_AA), returnValue, dretUsed, topLevel,
      unwrap(additionalArg), eunwrap(typeInfo, cast<Function>(unwrap(todiff))),
      uncacheable_args, eunwrap(augmented), AtomicAdd, PostOpt));
}
EnzymeAugmentedReturnPtr EnzymeCreateAugmentedPrimal(
    LLVMValueRef todiff, CDIFFE_TYPE retType, CDIFFE_TYPE *constant_args,
    size_t constant_args_size, LLVMTargetLibraryInfoRef TLI,
    EnzymeTypeAnalysisRef TA, EnzymeAAResultsRef global_AA, bool returnUsed,
    CFnTypeInfo typeInfo, bool *_uncacheable_args, size_t uncacheable_args_size,
    bool forceAnonymousTape, bool AtomicAdd, bool PostOpt) {

  std::vector<DIFFE_TYPE> nconstant_args((DIFFE_TYPE *)constant_args,
                                         (DIFFE_TYPE *)constant_args +
                                             constant_args_size);
  std::map<llvm::Argument *, bool> uncacheable_args;
  size_t argnum = 0;
  for (auto &arg : cast<Function>(unwrap(todiff))->args()) {
    assert(argnum < uncacheable_args_size);
    uncacheable_args[&arg] = _uncacheable_args[argnum];
    argnum++;
  }
  auto TLIpp = eunwrap(TLI);
  return ewrap(CreateAugmentedPrimal(
      cast<Function>(unwrap(todiff)), (DIFFE_TYPE)retType, nconstant_args,
      TLIpp, eunwrap(TA), eunwrap(global_AA), returnUsed,
      eunwrap(typeInfo, cast<Function>(unwrap(todiff))), uncacheable_args,
      forceAnonymousTape, AtomicAdd, PostOpt));
}
}