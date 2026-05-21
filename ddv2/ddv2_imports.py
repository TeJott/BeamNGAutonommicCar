"""
Pre-import stubs for NAVSIM/nuPlan to isolate the DDV2 model.
Import this BEFORE importing anything from navsim.
"""
import sys
import types
import numpy as np
from enum import IntEnum


# --- Enums ---
class StateSE2Index(IntEnum):
    X = 0
    Y = 1
    HEADING = 2

    @classmethod
    def size(cls):
        return len(cls)


class SE2Index(IntEnum):
    X = 0
    Y = 1
    HEADING = 2


class BoundingBox2DIndex(IntEnum):
    X = 0
    Y = 1
    HEADING = 2
    LENGTH = 3
    WIDTH = 4

    @classmethod
    def size(cls):
        return len(cls)


class PDMSimulationIndex(IntEnum):
    PROGRESS = 0
    TTC = 1
    COMFORT = 2
    DRIVABLE_AREA = 3
    SPEED_LIMIT = 4
    DIRECTION = 5


class MultiMetricIndex(IntEnum):
    NC = 0
    NO_COLLISION = 0
    EP = 1
    DAC = 2
    DRIVABLE_AREA = 2
    TTC = 3
    C = 4


class WeightedMetricIndex(IntEnum):
    PROGRESS = 0
    TTC = 1
    COMFORTABLE = 2
    DRIVING_DIRECTION = 3
    WEIGHTED = 4


class SemanticMapLayer(IntEnum):
    LANE = 0
    LANE_CONNECTOR = 1
    INTERSECTION = 2
    STOP_LINE = 3
    CROSSWALK = 4
    WALKWAYS = 5
    DRIVABLE_AREA = 6
    CARPARK_AREA = 7


class TrackedObjectType(IntEnum):
    VEHICLE = 0
    PEDESTRIAN = 1
    BICYCLE = 2
    GENERIC_OBJECT = 3
    EGO = 4
    BARRIER = 5
    TRAFFIC_CONE = 6
    CZONE_SIGN = 7


# --- Stub classes ---
class TrajectorySampling:
    def __init__(self, time_horizon=4.0, interval_length=0.5, num_poses=None, **kwargs):
        if num_poses is not None:
            self.time_horizon = num_poses * interval_length
        else:
            self.time_horizon = time_horizon
        self.interval_length = interval_length

    @property
    def num_poses(self):
        return int(self.time_horizon / self.interval_length)


class StateSE2:
    @staticmethod
    def deserialize(pose):
        return StateSE2()


class TimePoint:
    def __init__(self, time_us):
        self.time_us = time_us

    @property
    def time_s(self):
        return self.time_us / 1e6


class EgoState:
    def __init__(self):
        self.rear_axle = StateSE2()
        self.time_point = TimePoint(0)
        self.car_footprint = _CarFootprint()


class _CarFootprint:
    vehicle_parameters = {}


class InterpolatedTrajectory:
    def __init__(self, states):
        self._states = states
        self.start_time = TimePoint(0)
        self.end_time = TimePoint(4000000)

    def get_state_at_times(self, time_points):
        return [EgoState() for _ in time_points]


class PDMResults:
    def __init__(self, no_at_fault_collisions, drivable_area_compliance,
                 ego_progress, time_to_collision_within_bound, comfort,
                 driving_direction_compliance, score):
        self.no_at_fault_collisions = no_at_fault_collisions
        self.drivable_area_compliance = drivable_area_compliance
        self.ego_progress = ego_progress
        self.time_to_collision_within_bound = time_to_collision_within_bound
        self.comfort = comfort
        self.driving_direction_compliance = driving_direction_compliance
        self.score = score


class MetricCache:
    def __init__(self):
        self.ego_state = EgoState()
        self.trajectory = InterpolatedTrajectory([])
        self.observation = None
        self.centerline = None
        self.route_lane_ids = None
        self.drivable_area_map = None


class Trajectory:
    def __init__(self, poses=None):
        self.poses = poses
        self.trajectory_sampling = TrajectorySampling()


class AgentInput:
    pass


class Scene:
    pass


class Annotations:
    pass


class AbstractMap:
    pass


class MapObject:
    pass


class OrientedBox:
    pass


class LidarPointCloud:
    pass


class PDMSimulator:
    def __init__(self, *a, **kw): pass
    def simulate(self, *a, **kw): return None
    def simulate_proposals(self, *a, **kw): return None


class PDMScorerConfig:
    def __init__(self, *a, **kw):
        self.progress_weight = 5.0
        self.ttc_weight = 5.0
        self.comfortable_weight = 2.0
        self.driving_direction_weight = 0.0
        self.driving_direction_horizon = 1.0
        self.driving_direction_compliance_threshold = 2.0
        self.driving_direction_violation_threshold = 6.0
        self.stopped_speed_threshold = 5e-03
        self.progress_distance_threshold = 5.0
        self.weighted_metrics_array = np.ones(5)


class PDMScorer:
    def __init__(self, *a, **kw):
        self._multi_metrics = np.zeros((5, 1))
        self._weighted_metrics = np.zeros((5, 1))
        self._progress_raw = np.zeros(1)
        self._config = PDMScorerConfig()

    def score(self, *a, **kw): return {}
    def score_proposals(self, *a, **kw): return np.ones(1)


class LossComputer:
    def __init__(self, *a, **kw): pass


# --- Stub functions ---
def get_maps_api(*a, **kw): return None


def convert_absolute_to_relative_se2_array(*a, **kw): return None


def pdm_path(): return "stub"


def relative_to_absolute_poses(origin, states):
    return states


def _get_fixed_timesteps(ego_state, time_horizon, interval_length):
    n = int(time_horizon / interval_length) + 1
    return [TimePoint(int(i * interval_length * 1e6)) for i in range(n)]


def _se2_vel_acc_to_ego_state(state, vel, acc, timestep, vehicle_params):
    return EgoState()


def ego_states_to_state_array(ego_states):
    n = len(ego_states)
    return np.zeros((n, 7))


# --- Pre-populate sys.modules to prevent import chains ---
import os as _os

# Ensure intermediate packages exist (only for modules that DON'T exist on disk)
_intermediate_packages = [
    'navsim', 'navsim.common',
    'navsim.evaluate',
    'navsim.planning', 'navsim.planning.simulation',
    'navsim.planning.simulation.planner', 'navsim.planning.simulation.planner.pdm_planner',
    'navsim.planning.simulation.planner.pdm_planner.utils',
    'navsim.planning.simulation.planner.pdm_planner.simulation',
    'navsim.planning.simulation.planner.pdm_planner.scoring',
    'navsim.planning.simulation.planner.pdm_planner.proposal',
    'navsim.planning.metric_caching',
    'nuplan', 'nuplan.common', 'nuplan.common.maps', 'nuplan.common.maps.nuplan_map',
    'nuplan.common.actor_state', 'nuplan.common.geometry',
    'nuplan.database', 'nuplan.database.utils', 'nuplan.database.utils.pointclouds',
    'nuplan.database.maps_db',
    'nuplan.planning', 'nuplan.planning.simulation', 'nuplan.planning.simulation.trajectory',
    'nuplan.planning.simulation.planner', 'nuplan.planning.simulation.planner.ml_planner',
]
_stub_root = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'DiffusionDriveV2')
for pkg_name in _intermediate_packages:
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__name__ = pkg_name
        pkg.__package__ = '.'.join(pkg_name.split('.')[:-1]) if '.' in pkg_name else ''
        # Check if real directory exists
        pkg_path = _os.path.join(_stub_root, *pkg_name.split('.'))
        if _os.path.isdir(pkg_path):
            pkg.__path__ = [pkg_path]
        else:
            pkg.__path__ = []
        sys.modules[pkg_name] = pkg

_stub_modules = {
    'navsim.common.enums': types.ModuleType('navsim.common.enums'),
    'navsim.common.dataclasses': types.ModuleType('navsim.common.dataclasses'),
    'navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums':
        types.ModuleType('navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums'),
    'navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils':
        types.ModuleType('navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils'),
    'navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator':
        types.ModuleType('navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator'),
    'navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer':
        types.ModuleType('navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer'),
    'navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation':
        types.ModuleType('navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation'),
    'navsim.planning.simulation.planner.pdm_planner.proposal.planner_proposal':
        types.ModuleType('navsim.planning.simulation.planner.pdm_planner.proposal.planner_proposal'),
    'nuplan.planning.simulation.trajectory.trajectory_sampling':
        types.ModuleType('nuplan.planning.simulation.trajectory.trajectory_sampling'),
    'nuplan.planning.simulation.trajectory.interpolated_trajectory':
        types.ModuleType('nuplan.planning.simulation.trajectory.interpolated_trajectory'),
    'nuplan.planning.simulation.planner.ml_planner.transform_utils':
        types.ModuleType('nuplan.planning.simulation.planner.ml_planner.transform_utils'),
    'nuplan.common.maps.abstract_map':
        types.ModuleType('nuplan.common.maps.abstract_map'),
    'nuplan.common.actor_state.tracked_objects_types':
        types.ModuleType('nuplan.common.actor_state.tracked_objects_types'),
    'nuplan.common.actor_state.state_representation':
        types.ModuleType('nuplan.common.actor_state.state_representation'),
    'nuplan.common.actor_state.ego_state':
        types.ModuleType('nuplan.common.actor_state.ego_state'),
    'nuplan.common.actor_state.oriented_box':
        types.ModuleType('nuplan.common.actor_state.oriented_box'),
    'nuplan.common.geometry.convert':
        types.ModuleType('nuplan.common.geometry.convert'),
    'nuplan.database.utils.pointclouds.lidar':
        types.ModuleType('nuplan.database.utils.pointclouds.lidar'),
    'nuplan.common.maps.nuplan_map.map_factory':
        types.ModuleType('nuplan.common.maps.nuplan_map.map_factory'),
    'nuplan.database.maps_db.gpkg_mapsdb':
        types.ModuleType('nuplan.database.maps_db.gpkg_mapsdb'),
    'navsim.evaluate.pdm_score':
        types.ModuleType('navsim.evaluate.pdm_score'),
    'navsim.planning.metric_caching.metric_cache':
        types.ModuleType('navsim.planning.metric_caching.metric_cache'),
    'navsim.agents.diffusiondrivev2.modules.multimodal_loss':
        types.ModuleType('navsim.agents.diffusiondrivev2.modules.multimodal_loss'),
    'navsim.agents.diffusiondrivev2.transfuser_features':
        types.ModuleType('navsim.agents.diffusiondrivev2.transfuser_features'),
}

# Populate each stub module
for name, mod in _stub_modules.items():
    mod.__name__ = name
    mod.__file__ = f'<stub:{name}>'
    mod.__path__ = []
    mod.__package__ = '.'.join(name.split('.')[:-1])
    sys.modules[name] = mod

# --- Assign enums to stub modules ---
_stub_modules['navsim.common.enums'].StateSE2Index = StateSE2Index

# --- Assign dataclass stubs ---
_stub_modules['navsim.common.dataclasses'].Trajectory = Trajectory
_stub_modules['navsim.common.dataclasses'].AgentInput = AgentInput
_stub_modules['navsim.common.dataclasses'].Scene = Scene
_stub_modules['navsim.common.dataclasses'].Annotations = Annotations
_stub_modules['navsim.common.dataclasses'].StateSE2 = StateSE2
_stub_modules['navsim.common.dataclasses'].PDMResults = PDMResults

# --- Assign PDM stubs ---
_stub_modules['navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums'].SE2Index = SE2Index
_stub_modules['navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums'].PDMSimulationIndex = PDMSimulationIndex
_stub_modules['navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums'].MultiMetricIndex = MultiMetricIndex
_stub_modules['navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums'].WeightedMetricIndex = WeightedMetricIndex
_stub_modules['navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums'].pdm_path = "stub"
_stub_modules['navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils'].convert_absolute_to_relative_se2_array = convert_absolute_to_relative_se2_array
_stub_modules['navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator'].PDMSimulator = PDMSimulator
_stub_modules['navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer'].PDMScorer = PDMScorer
_stub_modules['navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer'].PDMScorerConfig = PDMScorerConfig
_stub_modules['navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation'].ego_states_to_state_array = ego_states_to_state_array

# --- Assign nuplan stubs ---
_stub_modules['nuplan.planning.simulation.trajectory.trajectory_sampling'].TrajectorySampling = TrajectorySampling
_stub_modules['nuplan.planning.simulation.trajectory.interpolated_trajectory'].InterpolatedTrajectory = InterpolatedTrajectory
_stub_modules['nuplan.planning.simulation.planner.ml_planner.transform_utils']._get_fixed_timesteps = _get_fixed_timesteps
_stub_modules['nuplan.planning.simulation.planner.ml_planner.transform_utils']._se2_vel_acc_to_ego_state = _se2_vel_acc_to_ego_state
_stub_modules['nuplan.common.maps.abstract_map'].SemanticMapLayer = SemanticMapLayer
_stub_modules['nuplan.common.maps.abstract_map'].AbstractMap = AbstractMap
_stub_modules['nuplan.common.maps.abstract_map'].MapObject = MapObject
_stub_modules['nuplan.common.actor_state.tracked_objects_types'].TrackedObjectType = TrackedObjectType
_stub_modules['nuplan.common.actor_state.state_representation'].StateSE2 = StateSE2
_stub_modules['nuplan.common.actor_state.state_representation'].TimePoint = TimePoint
_stub_modules['nuplan.common.actor_state.ego_state'].EgoState = EgoState
_stub_modules['nuplan.common.actor_state.oriented_box'].OrientedBox = OrientedBox
_stub_modules['nuplan.common.geometry.convert'].relative_to_absolute_poses = relative_to_absolute_poses
_stub_modules['nuplan.database.utils.pointclouds.lidar'].LidarPointCloud = LidarPointCloud
_stub_modules['nuplan.common.maps.nuplan_map.map_factory'].get_maps_api = get_maps_api
_stub_modules['nuplan.database.maps_db.gpkg_mapsdb'].MAP_LOCATIONS = {}

# --- Assign NAVSIM stubs ---
_stub_modules['navsim.evaluate.pdm_score'].pdm_score = lambda *a, **kw: PDMResults(True, True, 1.0, 1.0, 1.0, 1.0, 1.0)
_stub_modules['navsim.evaluate.pdm_score'].pdm_score_para = lambda *a, **kw: (None, None)
_stub_modules['navsim.planning.metric_caching.metric_cache'].MetricCache = MetricCache
_stub_modules['navsim.agents.diffusiondrivev2.transfuser_features'].BoundingBox2DIndex = BoundingBox2DIndex
_stub_modules['navsim.agents.diffusiondrivev2.modules.multimodal_loss'].LossComputer = LossComputer
