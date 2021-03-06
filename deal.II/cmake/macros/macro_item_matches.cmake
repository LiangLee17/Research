## ---------------------------------------------------------------------
## $Id: macro_item_matches.cmake 31527 2013-11-03 09:58:45Z maier $
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
# A small macro to test whether a given list contains an element.
#
# Usage:
#     ITEM_MATCHES(var regex list)
#
# var is set to true if list contains an item that matches regex.
#

MACRO(ITEM_MATCHES _var _regex)
  SET(${_var})
  FOREACH (_item ${ARGN})
    IF("${_item}" MATCHES ${_regex})
      SET(${_var} TRUE)
      BREAK()
    ENDIF()
  ENDFOREACH()
ENDMACRO()

