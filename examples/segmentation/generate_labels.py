from typing import List, Optional, Sequence, Tuple
from pathlib import Path
import click
from matplotlib.colors import ListedColormap
import numpy as np
import open3d as o3d
import cv2
from sklearn.cluster import DBSCAN
import imageio
from choixe.configurations import XConfig
from matplotlib import cm


class Segmenter(object):
    def __init__(
        self, image: np.ndarray, depth: np.ndarray, camera: np.ndarray, plane=None, f=1
    ):
        """Segment objects in image using the point cloud generated
        with depth and camera

        :param image: input image [H x W] or [H x W x 3]
        :type image: np.ndarray
        :param depth: input depth [H x W]
        :type depth: np.ndarray
        :param camera: camera matrix [3 x 3]
        :type camera: np.ndarray
        """

        self.image = image
        self.depth = depth
        self.camera = camera
        self._plane = None
        self.plane = plane
        self._f = f

    @property
    def plane(self):
        return self._plane

    @plane.setter
    def plane(self, v):
        self._plane = v

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, v):
        self._f = v

    def _load_pointclouds(
        self,
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """Loads the points cloud

        :return: point cloud with valid points only, point cloud with all points
        :rtype: Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]
        """

        h, w = self.depth.shape
        f = self._f
        self.downscaled_img = cv2.resize(self.image, (w // f, h // f))
        self.downscaled_depth = cv2.resize(
            self.depth, (w // f, h // f), interpolation=cv2.INTER_NEAREST
        )
        image_o3d = o3d.geometry.Image(self.downscaled_img)
        depth_o3d = o3d.geometry.Image(self.downscaled_depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            image_o3d, depth_o3d
        )
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
        intrinsics.height = h // f
        intrinsics.width = w // f
        mat = np.array(self.camera)
        mat //= f
        mat[2, 2] = 1.0
        intrinsics.intrinsic_matrix = mat.tolist()

        pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        pc_full = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsics, project_valid_depth_only=False
        )
        return pc, pc_full

    def _filter_pointcloud(
        self,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
        offset: float,
        offset_up: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filters the pointcloud by keeping only the points above the estimated plane

        :param distance_threshold: RANSAC plane estimation, max distance a point
        can be from the plane model, and still be considered an inlier
        :type distance_threshold: float
        :param ransac_n: RANSAC plane estimation, number of initial points to be
        considered inliers in each iteration
        :type ransac_n: int
        :param num_iterations: RANSAC plane estimation, number of iterations
        :type num_iterations: int
        :param offset: remove points that are closer to the plane than specified
        :type offset: float
        :param offset_up: remove points that are further away to the plane than specified
        :type offset_up: float
        :return: (depth mask of valid points, inliers indices)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        self.plane, inliers = self.pc.segment_plane(
            distance_threshold, ransac_n, num_iterations
        )

        pc_numpy = np.array(self.pc.points)
        a, b, c, d = self.plane
        distances = np.sum(pc_numpy * [a, b, c], axis=1) + d
        filtered = pc_numpy[np.logical_and(distances < -offset, distances > -offset_up)]
        self.pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(filtered))

        pc_full_numpy = np.array(self.pc_full.points)
        distances_full = np.sum(pc_full_numpy * [a, b, c], axis=1) + d
        plane_mask1 = distances_full < -offset
        plane_mask2 = distances_full > -offset_up
        plane_mask = np.logical_and(plane_mask1, plane_mask2)
        plane_mask = np.reshape(plane_mask, self.downscaled_depth.shape)

        return plane_mask, inliers

    def _cluster_pointcloud(self, eps: float, min_points: int) -> np.ndarray:
        """Clusters the points of the pointcloud

        :param eps: DBSCAN objects cluestering, density parameter that is used
        to find neighbouring points
        :type eps: float
        :param min_points: DBSCAN objects cluestering, minimum number of points
        to form a cluster
        :type min_points: int
        :return: labels for each point of the pointcloud
        :rtype: np.ndarray
        """

        samples = np.array(self.pc.points)[:, :3]
        if samples.size == 0:
            return np.array([])
        clustering = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1)
        clustering.fit(samples)
        clustering.labels_
        return clustering.labels_

    def _generate_instance_masks(
        self, valid_depth_mask: np.ndarray, labels: np.ndarray
    ) -> Sequence[np.ndarray]:
        """Generates instance masks given the depth mask of valid points
        and the labels for each valid point

        :param valid_depth_mask: depth mask of valid points [H x W]
        :type valid_depth_mask: np.ndarray
        :param labels: labels for each valid point
        :type labels: np.ndarray
        :return: instance masks
        :rtype: Sequence[np.ndarray]
        """

        # outliers (-1) will have the same label of background (0)
        labels += 1
        segmentation_mask = valid_depth_mask.astype(np.int32)
        segmentation_mask[segmentation_mask == 1] = labels
        # discard background label
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]
        masks = []
        for label in unique_labels:
            mask = segmentation_mask.copy()
            mask[np.where(mask != label)] = 0
            mask[np.where(mask == label)] = 1
            masks.append(mask)
        return masks

    def _densify_instance_mask(self, mask: np.ndarray) -> np.ndarray:
        """Densifies the instance mask

        :param mask: instance mask [H x W]
        :type mask: np.ndarray
        :return: dense instance mask [H x W]
        :rtype: np.ndarray
        """

        points = np.where(mask == 1)
        points = np.stack([points[1], points[0]], axis=1)
        hull = cv2.convexHull(points)
        dense_mask = cv2.fillPoly(mask.copy(), [hull], [1, 1, 1])
        return dense_mask

    def _merge_instance_masks(self, masks: Sequence[np.ndarray]) -> np.ndarray:
        """Merges the instance masks into segmentation mask

        :param masks: instance masks [H x W]
        :type masks: Sequence[np.ndarray]
        :return: segmentation mask [H x W]
        :rtype: np.ndarray
        """

        instance_masks = []
        for i, mask in enumerate(masks, start=1):
            mask = mask.copy()
            mask[mask == 1] = i
            instance_masks.append(mask)

        instance_masks = np.stack(instance_masks, axis=2)
        merged_mask = np.amax(instance_masks, axis=2).astype(np.uint8)
        return merged_mask

    def _color_segmentation_mask(
        self, mask: np.ndarray, map_type: Optional[str] = None
    ) -> np.ndarray:
        """Colors the segmentation mask

        :param mask: segmentation mask [H x W]
        :type mask: np.ndarray
        :param map_type: map type (e.g. 'objects')
        :type map_type: Optional[str]
        :return: colored segmentation mask [H x W x 3]
        :rtype: np.ndarray
        """

        if map_type is None:
            map_type = "objects"

        if map_type == "objects":
            colors = np.array(
                [
                    [66, 66, 66],
                    [239, 83, 80],
                    [66, 165, 245],
                    [76, 175, 80],
                    [255, 235, 59],
                    [156, 39, 176],
                ]
            )

        elif map_type == "plane":
            colors = np.array(
                [
                    [239, 83, 80],
                    [66, 66, 66],
                ]
            )

        colors = colors / 255.0
        n_colors = colors.shape[0]

        cmap = ListedColormap(colors)
        mask = mask.astype(np.float32)
        foreground = np.expand_dims(mask >= 0, -1)
        mask = (mask % n_colors) / n_colors
        mask = cmap(mask)[:, :, :3]
        mask = (mask * foreground * 255).astype(np.uint8)
        return mask

    def _crop_bounding_rectangle(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Crops an image and the corrisponding mask with the
        minimum bounding rectangle around the mask

        :param image: input image [H x W] or [H x W x C]
        :type image: np.ndarray
        :param mask: input mask [H x W]
        :type mask: np.ndarray
        :return: cropped image [H x W] or [H x W x C], cropped mask [H x W]
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        mask = (mask * 255).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask)
        cropped_image = image[y : y + h, x : x + w]
        cropped_mask = mask[y : y + h, x : x + w]
        return cropped_image, cropped_mask, [x, y, w, h]

    def segment(
        self,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
        offset: float,
        offset_up: float,
        eps: float,
        min_points: int,
        min_area: int,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray]]:
        """Creates a segmentation mask of the objects

        :param distance_threshold: RANSAC plane estimation, max distance a point
        can be from the plane model, and still be considered an inlier
        :type distance_threshold: float
        :param ransac_n:  RANSAC plane estimation, number of initial points to be
        considered inliers in each iteration
        :type ransac_n: int
        :param num_iterations: RANSAC plane estimation, number of iterations
        :type num_iterations: int
        :param offset: remove points that are closer to the plane than specified
        :type offset: float
        :param offset_up: remove points that are further away to the plane than specified
        :type offset_up: float
        :param eps: DBSCAN objects cluestering, density parameter that is used
        to find neighbouring points
        :type eps: float
        :param min_points: DBSCAN objects cluestering, minimum number of points
        to form a cluster
        :type min_points: int
        :param min_area: Minimum mask area relative to image size
        :type min_area: float
        :return: segmentation mask, colored segmentation mask, crops
        :rtype: dict
        """

        H, W = self.depth.shape
        self.pc, self.pc_full = self._load_pointclouds()
        valid_depth_mask = self.downscaled_depth > 0
        plane_mask, plane_inliers = self._filter_pointcloud(
            distance_threshold, ransac_n, num_iterations, offset, offset_up
        )

        valid_depth_mask *= plane_mask
        labels = self._cluster_pointcloud(eps, min_points)
        seg_masks = self._generate_instance_masks(valid_depth_mask, labels)
        seg_masks = [self._densify_instance_mask(x) for x in seg_masks]
        resized_seg_masks = [
            cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST) for x in seg_masks
        ]

        def area_check(mask):
            return ((mask > 0).sum() / (H * W)) > min_area

        valid_idxs = [area_check(x) for x in resized_seg_masks]
        crops = [
            self._crop_bounding_rectangle(self.image, x)
            for i, x in enumerate(resized_seg_masks)
            if valid_idxs[i]
        ]
        seg_masks = [x for i, x in enumerate(seg_masks) if valid_idxs[i]]
        resized_seg_masks = [
            x for i, x in enumerate(resized_seg_masks) if valid_idxs[i]
        ]

        if len(seg_masks) == 0:
            final_mask = np.zeros_like(valid_depth_mask).astype(np.uint8)
        else:
            final_mask = self._merge_instance_masks(seg_masks)
        colored_final_mask = self._color_segmentation_mask(
            final_mask, map_type="objects"
        )

        plane_mask = final_mask.copy()
        plane_mask[plane_mask > 0] = 1
        colored_plane_mask = self._color_segmentation_mask(plane_mask, map_type="plane")

        final_mask = cv2.resize(final_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        colored_final_mask = cv2.resize(
            colored_final_mask, (W, H), interpolation=cv2.INTER_NEAREST
        )
        colored_plane_mask = cv2.resize(
            colored_plane_mask, (W, H), interpolation=cv2.INTER_NEAREST
        )

        return {
            "final_mask": final_mask,
            "colored_final_mask": colored_final_mask,
            "colored_plane_mask": colored_plane_mask,
            "crops": crops,
            "full_point_cloud": self.pc_full,
            "plane_inliers": plane_inliers,
        }


def create_2d_stacked_image(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    plane_mask: np.ndarray,
    bounding_boxes: List[float],
    alpha: float = 0.5,
) -> np.ndarray:

    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    blended = (alpha * segmentation_mask + (1 - alpha) * image).astype(np.uint8)

    for bbox in bounding_boxes:
        x, y, w, h = bbox
        cv2.rectangle(blended, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)

    output = np.concatenate([image, plane_mask, segmentation_mask, blended], axis=1)
    return output


def create_3d_visualizer(
    point_cloud: o3d.geometry.PointCloud,
    inliers: Sequence[int] = None,
    visualizer_size: Sequence[int] = [800, 800],
    plane_color: Sequence[float] = [1.0, 0.0, 0.0],
) -> o3d.visualization.Visualizer:

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        width=visualizer_size[0], height=visualizer_size[1], left=0, top=0
    )

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=np.array([0, 0, 0])
    )
    if inliers is None:
        o3d.visualization.draw_geometries([point_cloud, mesh])
        vis.add_geometry(point_cloud)
        vis.add_geometry(mesh)
    else:
        inlier_cloud = point_cloud.select_by_index(inliers)
        inlier_cloud.paint_uniform_color(plane_color)
        outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
        vis.add_geometry(inlier_cloud)
        vis.add_geometry(outlier_cloud)
        vis.add_geometry(mesh)

    # Rotate view around target point cloud
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(1)
    view_control.set_lookat(
        np.array(point_cloud.points).mean(0)
    )  # LookAt Point Cloud centroid
    view_control.rotate(visualizer_size[0] / 4.0, -visualizer_size[1] / 4.0)

    return vis


@click.command("generate_labels")
@click.option("--data_folder", type=Path, required=True, help="Data folder path")
@click.option("-f", "--scale_factor", default=2, type=int, help="Scale factor")
@click.option(
    "-d",
    "--distance_threshold",
    type=float,
    default=0.01,
    help="RANSAC Plane - Max distance a point can be from the plane model, and still be considered an inlier",
)
@click.option(
    "-n",
    "--ransac_n",
    type=int,
    default=3,
    help="RANSAC Plane - Number of initial points to be considered inliers in each iteration",
)
@click.option(
    "-iter",
    "--num_iterations",
    type=int,
    default=1000,
    help="RANSAC Plane - Number of iterations",
)
@click.option(
    "--offset",
    type=float,
    default=0.01,
    help="Remove points that are closer to the plane than specified",
)
@click.option(
    "--offset_up",
    type=float,
    default=0.1,
    help="Remove points that are further away to the plane than specified",
)
@click.option(
    "--eps",
    type=float,
    default=0.01,
    help="DBSCAN - Density parameter that is used to find neighbouring points",
)
@click.option(
    "--min_points",
    type=int,
    default=50,
    help="DBSCAN - Minimum number of points to form a cluster",
)
@click.option(
    "--min_area",
    type=float,
    default=0.01,
    help="Minimum mask area relative to image size",
)
def generate_labels(
    data_folder,
    scale_factor,
    distance_threshold,
    ransac_n,
    num_iterations,
    offset,
    offset_up,
    eps,
    min_points,
    min_area,
):

    VISUALIZER_SIZE = [800, 800]
    try:
        from screeninfo import get_monitors

        max_height = int(get_monitors()[0].height)
        max_width = int(get_monitors()[0].width * 0.48)
        VISUALIZER_SIZE = [max_width, max_height]
    except:
        pass

    # Check for filenames
    image_filename = data_folder / "image.png"
    disparity_filename = data_folder / "disparity.png"
    calibration_filename = data_folder / "calibration.yml"

    for _filename in [image_filename, disparity_filename, calibration_filename]:
        assert _filename.exists(), f"Filename: {str(_filename)} not found!"

    # Load camera calibration data
    calibration = XConfig(calibration_filename)
    camera_matrix = calibration["camera_matrix"]["center"]
    baseline = calibration["baseline"]["center_right"]
    focal = camera_matrix[0][0]

    # Load image/disparity
    image = np.array(imageio.imread(str(image_filename)))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    disparity = np.array(imageio.imread(str(disparity_filename)))

    # Compute depth from disparity
    depth = (baseline * focal / disparity * 1000).astype(np.float32)

    # Segmentation Pipeline
    segmenter = Segmenter(image, depth, camera_matrix, f=scale_factor)
    segmentation_output = segmenter.segment(
        distance_threshold,
        ransac_n,
        num_iterations,
        offset,
        offset_up,
        eps,
        min_points,
        min_area,
    )

    # Visualization loop control
    visualization_running = True

    def stop_visualization(event=None):
        nonlocal visualization_running
        visualization_running = False

    # Visualize 3D PointCloud with plane segmentation
    visualizer_3d = create_3d_visualizer(
        point_cloud=segmentation_output["full_point_cloud"],
        inliers=segmentation_output["plane_inliers"],
        visualizer_size=VISUALIZER_SIZE,
    )
    visualizer_3d.register_key_callback(81, stop_visualization)

    # Visualize 2D Final Output
    colored_seg_mask = segmentation_output["colored_final_mask"]
    plane_seg_mask = segmentation_output["colored_plane_mask"]
    crops = segmentation_output["crops"]
    _, _, bounding_boxes = zip(*crops)
    out = create_2d_stacked_image(
        image, colored_seg_mask, plane_seg_mask, bounding_boxes
    )
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", VISUALIZER_SIZE[0], VISUALIZER_SIZE[1])
    cv2.moveWindow("output", int(VISUALIZER_SIZE[0] * 1.1), 0)

    # Visualization loop
    while visualization_running:
        visualizer_3d.poll_events()
        visualizer_3d.update_renderer()
        cv2.imshow("output", out)
        c = cv2.waitKey(10)
        if ord("q") == c:
            stop_visualization()

    # teardown visualizers
    visualizer_3d.destroy_window()
    cv2.destroyWindow("output")


if __name__ == "__main__":
    generate_labels()
