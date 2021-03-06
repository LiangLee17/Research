## ---------------------------------------------------------------------
## $Id: CMakeLists.txt 31861 2013-12-03 17:45:43Z maier $
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

IF(DEAL_II_COMPONENT_EXAMPLES)
  MESSAGE(STATUS "Setup examples")

  INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${DEAL_II_EXAMPLES_RELDIR}
    COMPONENT examples
    FILES_MATCHING
    #
    # Exclude folder structures: doc, doxygen, CMakeFiles,...
    #
    PATTERN "CMakeFiles*" EXCLUDE
    PATTERN "doc*" EXCLUDE
    PATTERN "doxygen*" EXCLUDE
    PATTERN ".svn*" EXCLUDE
    #
    # Glob Includes:
    #
    PATTERN "*.cc"
    PATTERN "*.prm"
    PATTERN "*.inp"
    PATTERN "step*/CMakeLists.txt"
    #
    # Special files:
    #
    PATTERN "output.reference.dat" # step-39
    PATTERN "postprocess.pl"       # step-39
    PATTERN "obstacle_file.pbm"    # step-42
    PATTERN "untitled.geo"         # step-49
    PATTERN "untitled.msh"         # step-49
    )

  MESSAGE(STATUS "Setup examples - Done")
ENDIF()
