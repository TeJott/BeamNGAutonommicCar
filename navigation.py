"""Navigation system using BeamNG road graph + A* pathfinding for turn-by-turn directions."""
import math
import heapq
import time


class NavigationSystem:
    """
    Route planning and turn-by-turn navigation using BeamNG's road graph.

    Modes:
    - Auto-routing: Uses NavigraphData to plan a route from current position
    - Manual waypoints: User-defined waypoint list

    The navigation output is a driving command:
    - 'follow_lane': continue straight, no maneuver needed
    - 'turn_left': upcoming left turn
    - 'turn_right': upcoming right turn
    - 'keep_straight': go straight at intersection (no turn)

    This command is consumed by DDV2's ego status vector to condition
    trajectory generation on the intended maneuver.
    """

    def __init__(self, bng=None):
        self.bng = bng
        self.graph = {}
        self.coords = {}
        self.route = []
        self.current_node = None
        self.destination_node = None
        self.manual_waypoints = []
        self.current_waypoint_idx = 0
        self.auto_distance = 500.0
        self.junction_angle_threshold = 30.0
        self.maneuver_lookahead = 3
        self.last_command = 'follow_lane'
        self.distance_to_maneuver = 999.0
        self._last_graph_fetch = 0
        self._graph_cache_ttl = 10.0

    def load_graph(self):
        """Fetch road graph from BeamNG's NavigraphData."""
        if self.bng is None:
            return False

        try:
            from beamngpy.tools import NavigraphData
            nav_data = NavigraphData.get_data(self.bng)
            self.graph = nav_data.get('graph', {})
            self.coords = nav_data.get('coords3d', {})
            self._last_graph_fetch = time.time()
            return len(self.graph) > 0
        except Exception as e:
            print(f"[NAV] Failed to load road graph: {e}")
            return False

    def set_destination(self, pos, max_distance=2000.0):
        """
        Plan route to a world coordinate.

        Args:
            pos: (x, y, z) world position
            max_distance: maximum graph search distance
        """
        if not self.graph:
            if not self.load_graph():
                print("[NAV] No road graph available")
                return False

        goal = self._find_nearest_node(pos)
        if goal is None:
            print("[NAV] Destination not on road graph")
            return False

        self.destination_node = goal
        return True

    def set_destination_auto(self, current_pos, distance=500.0):
        """
        Automatically plan a route forward along the road.

        Walks the graph forward from the current position to find
        a node approximately `distance` meters ahead.

        Args:
            current_pos: (x, y, z) current vehicle position
            distance: target route length in meters
        """
        if not self.graph:
            if not self.load_graph():
                return False

        self.auto_distance = distance
        start = self._find_nearest_node(current_pos)
        if start is None:
            return False

        goal = self._walk_forward(start, distance)
        if goal is None or goal == start:
            return False

        self.destination_node = goal
        self.route = self._astar(start, goal)
        return len(self.route) > 0

    def set_manual_waypoints(self, waypoints):
        """Set manual waypoint list for navigation."""
        self.manual_waypoints = list(waypoints)
        self.current_waypoint_idx = 0
        self.route = []
        self.destination_node = None

    def update(self, current_pos):
        """
        Update navigation state. Call every frame.

        Args:
            current_pos: (x, y, z) vehicle world position

        Returns:
            (command, distance_m): driving command and distance to maneuver
        """
        if self.manual_waypoints:
            return self._update_manual(current_pos)
        elif self.destination_node is not None:
            return self._update_auto(current_pos)
        else:
            return 'follow_lane', 999.0

    def get_current_command(self):
        """Get the last computed driving command."""
        return self.last_command, self.distance_to_maneuver

    def _update_manual(self, current_pos):
        """Follow manual waypoints."""
        if self.current_waypoint_idx >= len(self.manual_waypoints):
            return 'follow_lane', 0.0

        target = self.manual_waypoints[self.current_waypoint_idx]
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 5.0:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < len(self.manual_waypoints):
                target = self.manual_waypoints[self.current_waypoint_idx]
                dx = target[0] - current_pos[0]
                dy = target[1] - current_pos[1]
                dist = math.sqrt(dx**2 + dy**2)

        angle = math.degrees(math.atan2(dy, dx))
        cmd = self._angle_to_command(angle)

        self.last_command = cmd
        self.distance_to_maneuver = dist
        return cmd, dist

    def _update_auto(self, current_pos):
        """Update route and compute driving command from graph route."""
        if not self.graph:
            return 'follow_lane', 999.0

        nearest = self._find_nearest_node(current_pos)
        if nearest is None:
            return 'follow_lane', 999.0

        self.current_node = nearest

        if not self.route or nearest not in self.route:
            self.route = self._astar(nearest, self.destination_node)
            if not self.route:
                return 'follow_lane', 999.0

        current_idx = self.route.index(nearest) if nearest in self.route else 0
        cmd, dist = self._detect_next_maneuver(current_idx)
        self.last_command = cmd
        self.distance_to_maneuver = dist
        return cmd, dist

    def _find_nearest_node(self, pos, max_dist=50.0):
        """Find the closest graph node to a world position."""
        if not self.coords:
            return None

        best_node = None
        best_dist = float('inf')

        for node_id, coord in self.coords.items():
            dx = coord[0] - pos[0]
            dy = coord[1] - pos[1]
            dz = coord[2] - pos[2]
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            if d < best_dist and d < max_dist:
                best_dist = d
                best_node = node_id

        return best_node

    def _walk_forward(self, start_node, target_distance):
        """
        Walk the graph forward from start_node until cumulative distance
        exceeds target_distance. Returns the farthest node reached.
        """
        if start_node not in self.graph:
            return None

        visited = {start_node}
        frontier = [(start_node, 0.0)]
        farthest_node = start_node
        farthest_dist = 0.0

        while frontier:
            current, cum_dist = frontier.pop(0)
            if cum_dist > farthest_dist:
                farthest_dist = cum_dist
                farthest_node = current

            if current not in self.graph:
                continue

            for neighbor, edge_dist in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_dist = cum_dist + edge_dist
                    if new_dist >= target_distance:
                        return neighbor
                    frontier.append((neighbor, new_dist))

        return farthest_node

    def _astar(self, start, goal):
        """A* shortest path on the road graph."""
        if start not in self.graph or goal not in self.graph:
            return []

        def heuristic(a, b):
            ca = self.coords.get(a)
            cb = self.coords.get(b)
            if ca is None or cb is None:
                return 0.0
            return math.sqrt((ca[0] - cb[0])**2 + (ca[1] - cb[1])**2 + (ca[2] - cb[2])**2)

        open_set = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        max_iterations = 5000

        for _ in range(max_iterations):
            if not open_set:
                break

            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor, dist in self.graph.get(current, []):
                tentative = g_score[current] + dist
                if tentative < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score = tentative + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return []

    def _detect_next_maneuver(self, current_idx):
        """
        Detect the next maneuver from the current route position.

        Looks ahead on the route for junctions where the bearing change
        exceeds the threshold. Classifies as turn_left, turn_right,
        or keep_straight.

        Returns:
            (command, distance_meters)
        """
        if current_idx >= len(self.route) - 1:
            return 'follow_lane', 999.0

        cum_dist = 0.0

        for i in range(current_idx, min(current_idx + 20, len(self.route) - 2)):
            a = self.route[i]
            b = self.route[i + 1]
            c = self.route[i + 2] if i + 2 < len(self.route) else b

            ca = self.coords.get(a)
            cb = self.coords.get(b)
            cc = self.coords.get(c)

            if ca is None or cb is None:
                continue

            edge_len = math.sqrt((cb[0] - ca[0])**2 + (cb[1] - ca[1])**2)
            cum_dist += edge_len

            if cc is None:
                continue

            angle1 = math.atan2(cb[1] - ca[1], cb[0] - ca[0])
            angle2 = math.atan2(cc[1] - cb[1], cc[0] - cb[0])

            bearing_change = math.degrees(angle2 - angle1)
            bearing_change = (bearing_change + 180) % 360 - 180

            if abs(bearing_change) > self.junction_angle_threshold:
                if bearing_change < -self.junction_angle_threshold:
                    return 'turn_right', cum_dist
                else:
                    return 'turn_left', cum_dist

            # Check if at a junction (node with degree > 2)
            degree = len(self.graph.get(b, []))
            if degree > 2:
                if bearing_change < -10:
                    return 'turn_right', cum_dist
                elif bearing_change > 10:
                    return 'turn_left', cum_dist
                else:
                    return 'keep_straight', cum_dist

        return 'follow_lane', cum_dist

    def _angle_to_command(self, angle_deg):
        """Convert world-space angle to driving command."""
        angle = (angle_deg + 180) % 360 - 180
        if angle < -30:
            return 'turn_left'
        elif angle > 30:
            return 'turn_right'
        else:
            return 'follow_lane'

    def compute_driving_command(self, heading_rad):
        """
        Compute driving command from vehicle heading and route.
        Used when we have a heading from vehicle state.

        Args:
            heading_rad: vehicle heading in radians

        Returns:
            command string
        """
        if self.manual_waypoints and self.current_waypoint_idx < len(self.manual_waypoints):
            target = self.manual_waypoints[self.current_waypoint_idx]
            target_angle = math.atan2(
                target[1] - 0,  # assuming current at origin
                target[0] - 0
            )
            rel_angle = (target_angle - heading_rad + math.pi) % (2 * math.pi) - math.pi
            return self._angle_to_command(math.degrees(rel_angle))

        return self.last_command
