## ---------------------------------------------------------------------
## $Id: FindTBB.cmake 31527 2013-11-03 09:58:45Z maier $
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
# Try to find the Threading Building Blocks library
#
# This module exports
#
#   TBB_LIBRARIES
#   TBB_INCLUDE_DIRS
#   TBB_WITH_DEBUGLIB
#   TBB_VERSION
#   TBB_VERSION_MAJOR
#   TBB_VERSION_MINOR
#

INCLUDE(FindPackageHandleStandardArgs)

SET_IF_EMPTY(TBB_DIR "$ENV{TBB_DIR}")

FIND_PATH(TBB_INCLUDE_DIR tbb/tbb_stddef.h
  HINTS
    ${TBB_DIR}
  PATH_SUFFIXES include include/tbb tbb
  )

FILE(GLOB _path ${TBB_DIR}/build/*_release)

FIND_LIBRARY(TBB_LIBRARY
  NAMES tbb
  HINTS
    ${_path}
    ${TBB_DIR}
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
  )

FILE(GLOB _path ${TBB_DIR}/build/*_debug)

FIND_LIBRARY(TBB_DEBUG_LIBRARY
  NAMES tbb_debug
  HINTS
    ${_path}
    ${TBB_DIR}
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
  )

FIND_PACKAGE_HANDLE_STANDARD_ARGS(TBB DEFAULT_MSG
  TBB_LIBRARY
  TBB_INCLUDE_DIR
  )

IF(NOT TBB_INCLUDE_DIR MATCHES "-NOTFOUND")
  FILE(STRINGS "${TBB_INCLUDE_DIR}/tbb/tbb_stddef.h" TBB_VERSION_MAJOR_STRING
    REGEX "#define.*TBB_VERSION_MAJOR")
  STRING(REGEX REPLACE "^.*TBB_VERSION_MAJOR.*([0-9]+).*" "\\1"
    TBB_VERSION_MAJOR "${TBB_VERSION_MAJOR_STRING}"
    )

  FILE(STRINGS "${TBB_INCLUDE_DIR}/tbb/tbb_stddef.h" TBB_VERSION_MINOR_STRING
    REGEX "#define.*TBB_VERSION_MINOR")
  STRING(REGEX REPLACE "^.*TBB_VERSION_MINOR.*([0-9]+).*" "\\1"
    TBB_VERSION_MINOR "${TBB_VERSION_MINOR_STRING}"
    )

  SET(TBB_VERSION
    "${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}"
    )
ENDIF()

MARK_AS_ADVANCED(
  TBB_LIBRARY
  TBB_DEBUG_LIBRARY
  TBB_INCLUDE_DIR
  )

IF(TBB_FOUND)

  IF(NOT TBB_DEBUG_LIBRARY MATCHES "-NOTFOUND")
    SET(TBB_WITH_DEBUGLIB TRUE)
    SET(TBB_LIBRARIES debug ${TBB_DEBUG_LIBRARY} optimized ${TBB_LIBRARY})
  ELSE()
    SET(TBB_LIBRARIES ${TBB_LIBRARY})
  ENDIF()

  SET(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})

  MARK_AS_ADVANCED(TBB_DIR)
ELSE()
  SET(TBB_DIR "" CACHE PATH
    "An optional hint to a TBB installation"
    )
ENDIF()

