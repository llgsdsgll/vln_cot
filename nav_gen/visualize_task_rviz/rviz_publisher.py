"""
Run in ROS2 environment:
  python3 rviz_publisher.py --data_dir ./extracted

Publishes to RViz:
  /task/pointcloud     - PointCloud2 (scene, cropped)
  /task/navmesh        - Marker (triangle mesh)
  /task/trajectory     - Marker (line strip)
  /task/targets        - MarkerArray (spheres + text)

Point cloud input:
  - pointcloud.pcd     - preferred
  - pointcloud.npy     - legacy fallback
"""
import argparse
import json
import os
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import struct

from pcd_io import read_pcd


FRAME_ID = 'map'


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='./extracted')
    p.add_argument('--once', action='store_true', help='Publish once and exit')
    return p.parse_args()


def make_header(stamp):
    h = Header()
    h.stamp = stamp
    h.frame_id = FRAME_ID
    return h


# habitat uses Y-up; RViz uses Z-up → swap Y and Z
def hab2ros(pts):
    """(N,3) habitat coords → ROS coords: x=x, y=-z, z=y"""
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = -pts[:, 2]
    out[:, 2] = pts[:, 1]
    return out


def build_pointcloud2(stamp, pts, colors=None):
    pts_ros = hab2ros(pts).astype(np.float32)
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12
    if colors is not None:
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
        point_step = 16
        rgb_packed = np.zeros(len(pts), dtype=np.float32)
        c = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        for i, (r, g, b) in enumerate(c):
            packed = struct.pack('BBBB', b, g, r, 0)
            rgb_packed[i] = struct.unpack('f', packed)[0]
        data = np.hstack([pts_ros, rgb_packed.reshape(-1, 1)])
    else:
        data = pts_ros

    msg = PointCloud2()
    msg.header = make_header(stamp)
    msg.height = 1
    msg.width = len(pts)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * len(pts)
    msg.data = data.astype(np.float32).tobytes()
    msg.is_dense = True
    return msg


def load_pointcloud(data_dir):
    pcd_path = os.path.join(data_dir, 'pointcloud.pcd')
    if os.path.exists(pcd_path):
        return read_pcd(pcd_path)

    npy_path = os.path.join(data_dir, 'pointcloud.npy')
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f'No point cloud found in {data_dir}: expected pointcloud.pcd or pointcloud.npy')

    points = np.load(npy_path)
    pc_color_path = os.path.join(data_dir, 'pointcloud_colors.npy')
    colors = np.load(pc_color_path) if os.path.exists(pc_color_path) else None
    return points, colors


def build_navmesh_marker(stamp, verts, faces):
    m = Marker()
    m.header = make_header(stamp)
    m.ns = 'navmesh'
    m.id = 0
    m.type = Marker.TRIANGLE_LIST
    m.action = Marker.ADD
    m.scale.x = m.scale.y = m.scale.z = 1.0
    m.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.35)

    verts_ros = hab2ros(verts)
    if faces is not None:
        for f in faces:
            for vi in f:
                p = Point()
                p.x, p.y, p.z = float(verts_ros[vi, 0]), float(verts_ros[vi, 1]), float(verts_ros[vi, 2])
                m.points.append(p)
    else:
        # no face info: render as points via separate marker
        pass
    return m


def build_navmesh_points_marker(stamp, verts):
    """Fallback when no face data: render navmesh as small spheres."""
    m = Marker()
    m.header = make_header(stamp)
    m.ns = 'navmesh_pts'
    m.id = 1
    m.type = Marker.POINTS
    m.action = Marker.ADD
    m.scale.x = m.scale.y = 0.05
    m.color = ColorRGBA(r=0.2, g=0.8, b=1.0, a=0.5)
    verts_ros = hab2ros(verts)
    for v in verts_ros:
        p = Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
        m.points.append(p)
    return m


def build_trajectory_marker(stamp, traj):
    m = Marker()
    m.header = make_header(stamp)
    m.ns = 'trajectory'
    m.id = 2
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = 0.05
    m.color = ColorRGBA(r=1.0, g=0.4, b=0.0, a=1.0)
    traj_ros = hab2ros(traj)
    for pt in traj_ros:
        m.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))
    return m


def build_target_markers(stamp, targets):
    ma = MarkerArray()
    for i, t in enumerate(targets):
        pos = np.array([t['position']], dtype=np.float32)
        pos_ros = hab2ros(pos)[0]

        # Sphere
        sphere = Marker()
        sphere.header = make_header(stamp)
        sphere.ns = 'targets'
        sphere.id = i * 2
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = Point(x=float(pos_ros[0]), y=float(pos_ros[1]), z=float(pos_ros[2]))
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
        sphere.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.9)
        ma.markers.append(sphere)

        # Text label
        text = Marker()
        text.header = make_header(stamp)
        text.ns = 'target_labels'
        text.id = i * 2 + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = Point(x=float(pos_ros[0]), y=float(pos_ros[1]), z=float(pos_ros[2]) + 0.4)
        text.scale.z = 0.25
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text = t['name']
        ma.markers.append(text)

    return ma


class TaskVisualizer(Node):
    def __init__(self, data_dir, exit_after_publish=False):
        super().__init__('task_visualizer')
        qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_pc    = self.create_publisher(PointCloud2,  '/task/pointcloud',  qos)
        self.pub_nav   = self.create_publisher(MarkerArray,  '/task/navmesh',     qos)
        self.pub_traj  = self.create_publisher(Marker,       '/task/trajectory',  qos)
        self.pub_tgt   = self.create_publisher(MarkerArray,  '/task/targets',     qos)

        self._data = self._load(data_dir)
        self._exit_after_publish = exit_after_publish
        self._published = False
        self._shutdown_timer = None
        # Publish once after startup so DDS discovery has a moment to finish.
        self._publish_timer = self.create_timer(0.2, self._publish_once)

    def _load(self, d):
        data = {}
        data['traj']   = np.load(os.path.join(d, 'trajectory.npy'))
        data['pc'], data['pc_col'] = load_pointcloud(d)
        nav_v_path     = os.path.join(d, 'navmesh_verts.npy')
        nav_f_path     = os.path.join(d, 'navmesh_faces.npy')
        data['nav_v']  = np.load(nav_v_path) if os.path.exists(nav_v_path) else None
        data['nav_f']  = np.load(nav_f_path) if os.path.exists(nav_f_path) else None
        with open(os.path.join(d, 'targets.json')) as f:
            data['targets'] = json.load(f)
        with open(os.path.join(d, 'meta.json')) as f:
            data['meta'] = json.load(f)
        return data

    def _publish_once(self):
        if self._published:
            return

        self._published = True
        self._publish_timer.cancel()
        self._publish()

        if self._exit_after_publish:
            self.get_logger().info('Published all markers once. Shutting down.')
            self._shutdown_timer = self.create_timer(0.1, self._shutdown_once)
        else:
            self.get_logger().info('Published all markers once (latched). Open RViz to visualize.')

    def _shutdown_once(self):
        if self._shutdown_timer is not None:
            self._shutdown_timer.cancel()
        self.destroy_node()
        rclpy.shutdown()

    def _publish(self):
        now = self.get_clock().now().to_msg()
        d = self._data

        self.get_logger().info(f"Task: {d['meta']['task_instruction']}")

        # Point cloud
        if len(d['pc']) > 0:
            self.pub_pc.publish(build_pointcloud2(now, d['pc'], d['pc_col']))

        # Navmesh
        nav_ma = MarkerArray()
        if d['nav_v'] is not None:
            if d['nav_f'] is not None and len(d['nav_f']) > 0:
                nav_ma.markers.append(build_navmesh_marker(now, d['nav_v'], d['nav_f']))
            else:
                nav_ma.markers.append(build_navmesh_points_marker(now, d['nav_v']))
        self.pub_nav.publish(nav_ma)

        # Trajectory
        if len(d['traj']) > 0:
            self.pub_traj.publish(build_trajectory_marker(now, d['traj']))

        # Targets
        self.pub_tgt.publish(build_target_markers(now, d['targets']))


def main():
    args = get_args()
    rclpy.init()
    node = TaskVisualizer(args.data_dir, exit_after_publish=args.once)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
