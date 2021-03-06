#!/bin/perl
## ---------------------------------------------------------------------
## $Id: adjust.pl 31681 2013-11-16 08:04:28Z maier $
##
## Copyright (C) 2012 - 2013 by the deal.II Authors
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

my $header=1;

print << 'EOT'
//----------------------------------------------------------------------
// $Id: adjust.pl 31681 2013-11-16 08:04:28Z maier $
//
// Copyright (C) 1998 - 2013 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
//----------------------------------------------------------------------
EOT
    ;

while(<>)
{
    $header=0 unless m/\/\//;
    next if $header;
    print;
}
