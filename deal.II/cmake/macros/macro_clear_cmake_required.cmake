## ---------------------------------------------------------------------
## $Id: macro_clear_cmake_required.cmake 31527 2013-11-03 09:58:45Z maier $
##
## Copyright (C) 2013 by the deal.II authors
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
# A small macro to clear the CMAKE_REQUIRED_* variables.
#
# Usage:
#     CLEAR_CMAKE_REQUIRED_FLAGS
#

MACRO(CLEAR_CMAKE_REQUIRED)
  SET(CMAKE_REQUIRED_FLAGS)
  SET(CMAKE_REQUIRED_INCLUDES)
  SET(CMAKE_REQUIRED_LIBRARIES)
ENDMACRO()

