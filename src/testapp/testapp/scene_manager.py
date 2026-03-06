#!/usr/bin/env python3
"""
Planning scene manager for MoveIt2.

Adds/removes collision objects (primitives or STL meshes) from the MoveIt2
planning scene so the planner routes around them automatically.

Usage — import and use inside any ROS2 node:

    from testapp.scene_manager import SceneManager

    mgr = SceneManager(node)

    # Primitive shapes
    mgr.add_box   ('table',    size=(0.6, 0.8, 0.05), position=(0.4,  0.0, 0.1))
    mgr.add_sphere('ball',     radius=0.05,            position=(0.3,  0.1, 0.35))
    mgr.add_cylinder('pole',   radius=0.02, height=0.4, position=(0.3, -0.15, 0.3))

    # STL mesh (binary or ASCII)
    mgr.add_mesh('my_object', '/path/to/obstacle.stl', position=(0.4, 0.1, 0.2))

    # Remove one or all
    mgr.remove('table')
    mgr.clear()
"""

import trimesh

from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from std_msgs.msg import Header


def _make_pose(position, orientation=(0.0, 0.0, 0.0, 1.0)) -> Pose:
    pose = Pose()
    pose.position    = Point(x=float(position[0]),
                             y=float(position[1]),
                             z=float(position[2]))
    pose.orientation = Quaternion(x=float(orientation[0]),
                                  y=float(orientation[1]),
                                  z=float(orientation[2]),
                                  w=float(orientation[3]))
    return pose


class SceneManager:
    """Thin wrapper around MoveIt2 CollisionObject publishing."""

    PLANNING_FRAME = 'world'

    def __init__(self, node):
        self._node = node
        self._pub  = node.create_publisher(CollisionObject, '/collision_object', 10)
        self._tracked: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_box(self, name: str, size: tuple, position: tuple,
                orientation=(0.0, 0.0, 0.0, 1.0)) -> None:
        """
        Add a box obstacle.

        Args:
            name:        Unique ID for this object.
            size:        (x, y, z) dimensions in metres.
            position:    (x, y, z) centre position in metres (world frame).
            orientation: (x, y, z, w) quaternion — default is upright.
        """
        primitive         = SolidPrimitive()
        primitive.type    = SolidPrimitive.BOX
        primitive.dimensions = [float(size[0]), float(size[1]), float(size[2])]
        self._publish_primitive(name, primitive, position, orientation)

    def add_sphere(self, name: str, radius: float, position: tuple) -> None:
        """
        Add a sphere obstacle.

        Args:
            name:     Unique ID for this object.
            radius:   Radius in metres.
            position: (x, y, z) centre position in metres (world frame).
        """
        primitive              = SolidPrimitive()
        primitive.type         = SolidPrimitive.SPHERE
        primitive.dimensions   = [float(radius)]
        self._publish_primitive(name, primitive, position)

    def add_cylinder(self, name: str, radius: float, height: float,
                     position: tuple, orientation=(0.0, 0.0, 0.0, 1.0)) -> None:
        """
        Add a cylinder obstacle. The cylinder axis is along Z.

        Args:
            name:        Unique ID for this object.
            radius:      Radius in metres.
            height:      Height in metres.
            position:    (x, y, z) centre position in metres (world frame).
            orientation: (x, y, z, w) quaternion — default is upright.
        """
        primitive            = SolidPrimitive()
        primitive.type       = SolidPrimitive.CYLINDER
        primitive.dimensions = [float(height), float(radius)]
        self._publish_primitive(name, primitive, position, orientation)

    def add_mesh(self, name: str, stl_path: str, position: tuple,
                 orientation=(0.0, 0.0, 0.0, 1.0), scale: float = 1.0) -> None:
        """
        Add a mesh obstacle loaded from an STL file (binary or ASCII).

        Args:
            name:        Unique ID for this object.
            stl_path:    Absolute path to the .stl file.
            position:    (x, y, z) position in metres (world frame).
            orientation: (x, y, z, w) quaternion.
            scale:       Uniform scale factor (default 1.0 = no scaling).
                         Useful if the STL is in mm (use scale=0.001).
        """
        try:
            loaded = trimesh.load(stl_path, force='mesh')
        except Exception as e:
            self._node.get_logger().error(f'Failed to load STL "{stl_path}": {e}')
            return

        if scale != 1.0:
            loaded.apply_scale(scale)

        mesh_msg = Mesh()
        for tri in loaded.faces:
            t = MeshTriangle()
            t.vertex_indices = [int(tri[0]), int(tri[1]), int(tri[2])]
            mesh_msg.triangles.append(t)
        for v in loaded.vertices:
            mesh_msg.vertices.append(Point(x=float(v[0]), y=float(v[1]), z=float(v[2])))

        obj                    = self._make_collision_object(name)
        obj.meshes             = [mesh_msg]
        obj.mesh_poses         = [_make_pose(position, orientation)]
        obj.operation          = CollisionObject.ADD
        self._publish(name, obj)
        self._node.get_logger().info(
            f'[SceneManager] Added mesh "{name}" from {stl_path} '
            f'({len(loaded.faces)} triangles)'
        )

    def remove(self, name: str) -> None:
        """Remove a single collision object by name."""
        obj           = self._make_collision_object(name)
        obj.operation = CollisionObject.REMOVE
        self._pub.publish(obj)
        self._tracked.discard(name)
        self._node.get_logger().info(f'[SceneManager] Removed "{name}"')

    def clear(self) -> None:
        """Remove all collision objects added by this manager."""
        for name in list(self._tracked):
            self.remove(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_collision_object(self, name: str) -> CollisionObject:
        obj         = CollisionObject()
        obj.header  = Header()
        obj.header.frame_id = self.PLANNING_FRAME
        obj.id      = name
        return obj

    def _publish_primitive(self, name: str, primitive: SolidPrimitive,
                           position: tuple,
                           orientation=(0.0, 0.0, 0.0, 1.0)) -> None:
        obj                    = self._make_collision_object(name)
        obj.primitives         = [primitive]
        obj.primitive_poses    = [_make_pose(position, orientation)]
        obj.operation          = CollisionObject.ADD
        self._publish(name, obj)
        kind = {SolidPrimitive.BOX: 'box',
                SolidPrimitive.SPHERE: 'sphere',
                SolidPrimitive.CYLINDER: 'cylinder'}.get(primitive.type, '?')
        self._node.get_logger().info(
            f'[SceneManager] Added {kind} "{name}" at {position}'
        )

    def _publish(self, name: str, obj: CollisionObject) -> None:
        self._pub.publish(obj)
        self._tracked.add(name)
