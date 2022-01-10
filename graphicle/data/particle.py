

# @attr.s
# class FourMomentum:
#     pmu: np.ndarray = attr.ib(
#             converter=utils.structure_pmu,
#             eq=attr.cmp_using(eq=np.array_equal),
#             )

#     @classmethod
#     def from_components(cls,
#                         x: np.ndarray, y: np.ndarray,
#                         z: np.ndarray, e: np.ndarray):
#         struc_pmu = utils.structure_pmu_components(x, y, z, e)
#         return cls(struc_pmu)

#     @property
#     def strip_names(self):
#         return utils.unstructure_pmu(self.pmu, dtype=REAL_TYPE)

#     @property
#     def mag(self):
#         pmu2 = self.strip_names(self.pmu) ** 2
#         return np.sqrt(np.sum(pmu2, axis=0))

#     @property
#     def pt(self):
#         return np.sqrt(self.pmu['x']**2 + self.pmu['y']**2)

#     @propety
#     def eta(self):
#         return np.arctanh(self.pmu['z'] / self.mag(self.pmu), axis=0)
            

