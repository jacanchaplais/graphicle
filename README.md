# graphicle

[![PyPI version](https://img.shields.io/pypi/v/graphicle.svg)](https://pypi.org/project/graphicle/)
[![Documentation](https://readthedocs.org/projects/graphicle/badge/?version=latest)](https://graphicle.readthedocs.io)
[![License](https://img.shields.io/pypi/l/graphicle)](https://raw.githubusercontent.com/jacanchaplais/graphicle/main/LICENSE.txt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Utilities for representing high energy physics data as graphs / networks.


## Installation
```
pip install graphicle
```

# Features

Object oriented interface to track-level particle data for collider physics,
with routines for constructing and performing calculations over
graph-structured data.

Provides data structures for:
* 4-momenta
* PDG codes
* Particle status codes
* Color codes
* Helicity / spin polarisation data
* COO adjacency lists (for graph-structured data)

```python3
>>> import graphicle as gcl

# query pdg records
>>> pdgs = gcl.PdgArray([1, 3, 6, -6, 25, 2212])
>>> pdgs.name
['d', 's', 't', 't~', 'H0', 'p'], dtype=object)
>>> pdgs.charge
array([-0.33333333, -0.33333333,  0.66666667, -0.66666667,  0.        ,
        1.        ])

# extract information from momentum data
>>> pmu_data
array([( 1.95057378e-02,  3.12923088e-02,  3.53556064e-01, 3.55473730e-01),
       ( 2.60116947e+01, -3.63466398e+00, -3.33718718e+00, 2.64755711e+01),
       ( 5.91884324e-05, -7.62144267e-06, -6.76385314e-06, 6.00591927e-05),
       ( 2.82881807e+01,  4.32224823e+00,  2.14691072e+02, 2.16589841e+02),
       (-8.73280642e-02, -6.48540201e-02,  3.73744945e-01, 6.28679140e-01),
       ( 1.06204871e-01,  5.78888984e-01, -1.44899819e+02, 1.44901081e+02)],
      dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('e', '<f8')])
>>> pmu = gcl.MomentumArray(pmu_data)
>>> pmu.pt
array([3.68738715e-02, 2.62644064e+01, 5.96771055e-05, 2.86164812e+01,
       1.08776076e-01, 5.88550704e-01])
>>> pmu.mass
array([-7.45058060e-09,  5.11000489e-04,  9.09494702e-13,  5.10991478e-04,
        4.93680000e-01,  1.39570000e-01])
>>> pmu.eta
array([ 2.95639434, -0.12672178, -0.11309956,  2.71277683,  1.94796328,
       -6.1992861 ])
>>> pmu.phi
array([ 1.01339184, -0.138833  , -0.12806107,  0.15162078, -2.5028134 ,
        1.38935084])

# calculate the inter-particle distances
>>> pmu.delta_R(pmu)
array([[0.        , 3.2913868 , 3.27485993, 0.89554388, 2.94501476,
        9.16339617],
       [3.2913868 , 0.        , 0.01736661, 2.85431528, 3.14526968,
        6.26189934],
       [3.27485993, 0.01736661, 0.        , 2.83968296, 3.14442819,
        6.27249595],
       [0.89554388, 2.85431528, 2.83968296, 0.        , 2.76241933,
        8.99760198],
       [2.94501476, 3.14526968, 3.14442819, 2.76241933, 0.        ,
        8.4908571 ],
       [9.16339617, 6.26189934, 6.27249595, 8.99760198, 8.4908571 ,
        0.        ]])
```

Graphicle really shines with its composite data structures. These can be used
to filter and query heterogeneous particle data records simultaneously, either
using user provided boolean masks, or `MaskArray` instances produced with
routines in the `select` module.
Additionally, routines in the `calculate` and `transform` modules take
composite data structures to standardise useful calculations which blends
multiple particle data records.

To see an example, let's generate a collision event using Pythia, wrapped
with `showerpipe`.


```python3
>>> from showerpipe.generator import PythiaGenerator
...
... lhe_path = "https://zenodo.org/record/6034610/files/unweighted_events.lhe.gz"
... gen = PythiaGenerator("pythia-settings.cmnd", lhe_path)
>>> for event in gen:
...     graph = gcl.Graphicle.from_event(event)
...     break

>>> graph.pdg
PdgArray(data=array([2212, 2212,   21, ...,   22,   22,   22], dtype=int32))
>>> graph.edges
array([(   0,   -1), (   0,   -2), (  -6,   -3), ..., (-635, 1211),
       (-636, 1212), (-636, 1213)], dtype=[('in', '<i4'), ('out', '<i4')])
# select all descendants of the W bosons from the hard process
>>> W_mask = gcl.select.hard_descendants(graph, {24})
>>> W_mask
MaskGroup(mask_arrays=["W+", "W-"], agg_op=OR)
# filter data record to get final state W+ boson descendants
>>> Wp_desc = graph[W_mask["W+"] & graph.final]
>>> Wp_desc.pdg
PdgArray(data=array([ 321, -211, -211,  321, -211, -321,  211,  211,  -13,   14,   22,
         22,  211, -211,   22,   22,   22,   22,   22,  211, -211,   22,
         22,   22,   22,  130,   22,   22], dtype=int32))
>>> Wp_desc
Graphicle(particles=ParticleSet(
PdgArray(data=array([ 321, -211, -211,  321, -211, -321,  211,  211,  -13,   14,   22,
         22,  211, -211,   22,   22,   22,   22,   22,  211, -211,   22,
         22,   22,   22,  130,   22,   22], dtype=int32)),
MomentumArray(data=array([(-1.41648688e+00, -2.6653416 , -2.25487483e-01, 3.06676466e+00),
       ( 5.26078595e-01,  0.11325339, -1.85115863e+00, 1.93283550e+00),
       ( 2.92112800e+00,  2.19611382, -9.04351574e+00, 9.75502749e+00),
       ( 1.70197168e+01,  9.65578074, -4.51506419e+01, 4.92110663e+01),
       (-5.70145778e-01, -1.02762625,  1.35915720e-01, 1.19123247e+00),
       (-1.70566595e-01,  0.02598637, -1.34183423e-01, 5.39901276e-01),
       (-1.80439204e-01, -0.51409054,  1.82537117e-01, 5.91309546e-01),
       ( 1.63182285e-01,  0.13788241, -3.17043212e-01, 4.06984277e-01),
       (-2.45719652e+00, -4.10607321,  3.31426006e-01, 4.79777648e+00),
       (-1.08820465e+00, -1.84333164, -1.69547133e-01, 2.14727900e+00),
       (-4.92718715e-01, -0.87998859,  1.11984849e-01, 1.01473753e+00),
       ( 8.90383374e-03, -0.01019132,  4.32869417e-04, 1.35398920e-02),
       (-6.11110402e-01, -0.74064239,  5.47809445e-02, 9.71847628e-01),
       (-2.13853648e-01, -0.34188095, -1.89837677e-01, 4.67048281e-01),
       (-3.57251890e-01, -0.42033772, -1.39634796e-01, 5.69043576e-01),
       (-2.41744268e-01,  0.16830106, -1.53611666e-02, 2.94960174e-01),
       (-8.27775995e-01, -0.4279882 ,  1.03575995e-01, 9.37611318e-01),
       (-3.44298782e-05,  0.14091286, -4.51929191e-02, 1.47982551e-01),
       ( 6.20276481e-02,  0.12552564, -1.96113732e-01, 2.40966203e-01),
       ( 6.32168629e+00,  4.5683574 , -1.69888394e+01, 1.86942171e+01),
       ( 8.77035615e-01,  0.4961944 , -2.38422385e+00, 2.59218122e+00),
       (-1.12781117e+00, -1.41626175, -6.02316244e-02, 1.81145887e+00),
       (-1.52146265e+00, -1.67738354, -3.45502640e-02, 2.26487480e+00),
       ( 1.82715744e+00,  0.28701504, -3.76239153e+00, 4.19243031e+00),
       ( 4.77818092e-01,  0.02881935, -8.63039360e-01, 9.86903046e-01),
       (-3.03560171e+00, -2.76703663,  9.57894838e-02, 4.13861822e+00),
       ( 8.99971241e-01,  0.6677899 , -2.26276823e+00, 2.52507657e+00),
       ( 1.42885287e+00,  0.86196369, -3.46387012e+00, 3.84486646e+00)],
      dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('e', '<f8')])),
ColorArray(data=array([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
       (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
       (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
       (0, 0), (0, 0), (0, 0), (0, 0)],
      dtype=[('color', '<i4'), ('anticolor', '<i4')])),
HelicityArray(data=array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
       9, 9, 9, 9, 9, 9], dtype=int16)),
StatusArray(data=array([83, 84, 84, 84, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91,
       91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91], dtype=int16)),
MaskArray(data=array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True]))
), adj=AdjacencyList(_data=array([(-343,  650), (-343,  651), (-343,  652), (-343,  653),
       (-345,  743), (-349,  744), (-349,  745), (-350,  746),
       (-344,  863), (-344,  864), (-346,  865), (-346,  866),
       (-347,  867), (-347,  868), (-347,  869), (-348,  870),
       (-348,  871), (-351,  872), (-351,  873), (-352,  874),
       (-352,  875), (-518, 1012), (-518, 1013), (-519, 1014),
       (-519, 1015), (-571, 1097), (-572, 1098), (-572, 1099)],
      dtype=[('in', '<i4'), ('out', '<i4')]), weights=array([], dtype=float64)))

# calculate the mass of the W boson from its final state constituents
>>> gcl.calculate.combined_mass(Wp_desc.pmu)
80.419002446
```

More information on the API is available in the
[documentation](https://graphicle.readthedocs.io)
