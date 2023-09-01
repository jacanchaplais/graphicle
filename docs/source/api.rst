.. py:module:: graphicle

graphicle
=========

The data structures and routines provided here enable the semantic
storage, querying, and manipulation of data from high energy physics
collision event records. This applies to both specific, homeogeneous
data from the event record, *eg.* PDG codes, and to composite,
heterogeneous objects comprising the whole record, *eg.* ``Graphicle``
objects.

The routines allow users to traverse ancestry using a topological
directed cyclic graph (DAG) representation, select specific regions
of the event, form clusters based on Monte-Carlo truth information,
and perform calculations, such as the mass of said clusters.

HEP calculation routines
------------------------
The following routines provide standard calculation utilities over
particle physics data.

.. python-apigen-group:: calculate

Data structures
---------------
Data structures to semantically handle records for particle collision
events. Interoperability with ``numpy`` API for non-composite objects
is built-in.

.. python-apigen-group:: datastructure

Data transformation routines
----------------------------
Provides functions to alter the representation of particle physics
data.

.. python-apigen-group:: transform

Matrix creation routines
------------------------
Functions to create matrices (mostly representing inter-particle
relationships) from particle physics data.

.. python-apigen-group:: matrix

Dataset filtering / masking
---------------------------
Algorithms to query event record, providing masks which select specific
regions of collision events.

.. python-apigen-group:: select

Exceptions and warnings
-------------------
Custom classes indicating to users specific issues relating to the unique
cross-over between high energy physics and graph algorithms.

.. python-apigen-group:: errors_warnings
