// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "triton/core/tritonserver.h"

// Suppress warnings in armnn headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma warning(push, 0)
#include <armnn/ArmNN.hpp>  // One-stop header for ArmNN
#pragma warning(pop)
#pragma GCC diagnostic pop

namespace triton { namespace backend { namespace armnn {

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

TRITONSERVER_DataType ConvertArmNNTypeToDataType(
    const armnn::DataType& ttype);
std::pair<bool, armnn::DataType> ConvertDataTypeToArmNNType(
    const TRITONSERVER_DataType dtype);
std::pair<bool, armnn::DataType> ModelConfigDataTypeToArmNNType(
    const std::string& data_type_str);

}}}  // namespace triton::backend::armnn