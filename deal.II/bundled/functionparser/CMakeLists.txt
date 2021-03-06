## ---------------------------------------------------------------------
## $Id: CMakeLists.txt 31682 2013-11-16 08:31:05Z maier $
##
## Copyright (C) 2012 - 2013 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------

#
# Remove -Wall and -pedantic from CMAKE_CXX_FLAGS (in directory scope) to
# avoid some annoying warnings...
#
STRIP_FLAG(CMAKE_CXX_FLAGS "-Wall")
STRIP_FLAG(CMAKE_CXX_FLAGS "-pedantic")

DEAL_II_ADD_LIBRARY(obj_functionparser OBJECT
  fparser.cc
  fpoptimizer.cc
  )

IF(DEAL_II_WITH_THREADS)
  DEAL_II_ADD_DEFINITIONS(obj_functionparser "FP_USE_THREAD_SAFE_EVAL")
ENDIF()
