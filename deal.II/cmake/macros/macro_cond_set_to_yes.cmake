## ---------------------------------------------------------------------
## $Id: macro_cond_set_to_yes.cmake 31527 2013-11-03 09:58:45Z maier $
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
# If bool is "true" (in a cmake fashion...), set variable to "yes",
# otherwise to "no".
#
# Usage:
#     COND_SET_TO_YES(bool variable)
#

MACRO(COND_SET_TO_YES _bool _variable)
  IF(${_bool})
    SET(${_variable} "yes")
  ELSE()
    SET(${_variable} "no")
  ENDIF()
ENDMACRO()

