from torch_pointcloud.utils.data import PointCloudDataLoader

from .modelnet import ModelNet10, ModelNet40, ModelNetNormalResampled
from .parislille3d import ParisLille3D
from .repeat import RepeatDataset
from .s3dis import S3DIS, S3DISHdf5
from .scannet import ScanNet, ScanNet20, ScanNet200
from .scanobjectnn import ScanObjectNN
from .semantic3d import Semantic3D
from .semantickitti import SemanticKITTI
from .shapenetpart import ShapeNetPart
from .sunrgbd import SunRGBD
from .toronto3d import Toronto3D
