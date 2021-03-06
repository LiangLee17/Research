## ---------------------------------------------------------------------
## $Id: configure_mumps.cmake 31644 2013-11-13 23:48:05Z maier $
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
# Configuration for the MUMPS library:
#

SET(FEATURE_MUMPS_DEPENDS DEAL_II_WITH_MPI DEAL_II_WITH_LAPACK)


#
# The user has to know the location of the MUMPS headers as well:
#
SET(MUMPS_ADD_TO_USER_INCLUDE_DIRS TRUE)


CONFIGURE_FEATURE(MUMPS)
