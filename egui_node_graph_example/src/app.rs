use std::{borrow::Cow, collections::HashMap};

use eframe::egui::{self, DragValue, TextStyle};
use egui_node_graph::*;

// ========= First, define your user data types =============

/// The NodeData holds a custom data struct inside each node. It's useful to
/// store additional information that doesn't live in parameters. For this
/// example, the node data stores the template (i.e. the "type") of the node.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct MyNodeData {
    template: MyNodeTemplate,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum DepthaiNode {
    ColorCamera,
    MonoCamera,
    ImageManip,
    VideoEncoder,

    NeuralNetwork,
    DetectionNetwork,
    MobileNetDetectionNetwork,
    MobileNetSpatialDetectionNetwork,
    YoloDetectionNetwork,
    YoloSpatialDetectionNetwork,
    SpatialDetectionNetwork,

    SPIIn,
    XLinkIn,

    SPIOut,
    XLinkOut,

    Script,

    StereoDepth,
    SpatialLocationCalculator,

    EdgeDetector,
    FeaureTracker,
    ObjectTracker,
    IMU,
}

impl DepthaiNode {
    fn name(&self) -> String {
        match self {
            Self::ColorCamera => "Color Camera".to_string(),
            Self::MonoCamera => "Mono Camera".to_string(),
            Self::ImageManip => "Image Manipulation".to_string(),
            Self::VideoEncoder => "Video Encoder".to_string(),

            Self::NeuralNetwork => "Neural Network".to_string(),
            Self::DetectionNetwork => "Detection Network".to_string(),
            Self::MobileNetDetectionNetwork => "MobileNet Detection Network".to_string(),
            Self::MobileNetSpatialDetectionNetwork => {
                "MobileNet Spatial Detection Network".to_string()
            }
            Self::YoloDetectionNetwork => "Yolo Detection Network".to_string(),
            Self::YoloSpatialDetectionNetwork => "Yolo Spatial Detection Network".to_string(),
            Self::SpatialDetectionNetwork => "Spatial Detection Network".to_string(),

            Self::SPIIn => "SPI In".to_string(),
            Self::XLinkIn => "XLink In".to_string(),

            Self::SPIOut => "SPI Out".to_string(),
            Self::XLinkOut => "XLink Out".to_string(),

            Self::Script => "Script".to_string(),

            Self::StereoDepth => "Stereo Depth".to_string(),
            Self::SpatialLocationCalculator => "Spatial Location Calculator".to_string(),

            Self::EdgeDetector => "Edge Detector".to_string(),
            Self::FeaureTracker => "Feature Tracker".to_string(),
            Self::ObjectTracker => "Object Tracker".to_string(),
            Self::IMU => "IMU".to_string(),
        }
    }
}

/// `DataType`s are what defines the possible range of connections when
/// attaching two ports together. The graph UI will make sure to not allow
/// attaching incompatible datatypes.
#[derive(Eq)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub enum MyDataType {
    Scalar,
    Vec2,

    Queue(DepthaiNode),
}

impl PartialEq for MyDataType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scalar, Self::Scalar) => true,
            (Self::Vec2, Self::Vec2) => true,
            (Self::Queue(_), Self::Queue(_)) => true,
            _ => false,
        }
    }
}

/// In the graph, input parameters can optionally have a constant value. This
/// value can be directly edited in a widget inside the node itself.
///
/// There will usually be a correspondence between DataTypes and ValueTypes. But
/// this library makes no attempt to check this consistency. For instance, it is
/// up to the user code in this example to make sure no parameter is created
/// with a DataType of Scalar and a ValueType of Vec2.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub enum MyValueType {
    Vec2 { value: egui::Vec2 },
    Scalar { value: f32 },

    Queue(DepthaiNode),
}

impl Default for MyValueType {
    fn default() -> Self {
        // NOTE: This is just a dummy `Default` implementation. The library
        // requires it to circumvent some internal borrow checker issues.
        Self::Scalar { value: 0.0 }
    }
}

impl MyValueType {
    /// Tries to downcast this value type to a vector
    pub fn try_to_vec2(self) -> anyhow::Result<egui::Vec2> {
        if let MyValueType::Vec2 { value } = self {
            Ok(value)
        } else {
            anyhow::bail!("Invalid cast from {:?} to vec2", self)
        }
    }

    /// Tries to downcast this value type to a scalar
    pub fn try_to_scalar(self) -> anyhow::Result<f32> {
        if let MyValueType::Scalar { value } = self {
            Ok(value)
        } else {
            anyhow::bail!("Invalid cast from {:?} to scalar", self)
        }
    }

    /// Tries to downcast this value type to a queue
    pub fn try_to_node(self) -> anyhow::Result<DepthaiNode> {
        if let MyValueType::Queue(node) = self {
            Ok(node)
        } else {
            anyhow::bail!("Invalid cast from {:?} to node", self)
        }
    }
}

/// NodeTemplate is a mechanism to define node templates. It's what the graph
/// will display in the "new node" popup. The user code needs to tell the
/// library how to convert a NodeTemplate into a Node.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub enum MyNodeTemplate {
    MakeScalar,
    AddScalar,
    SubtractScalar,
    MakeVector,
    AddVector,
    SubtractVector,
    VectorTimesScalar,

    CreateColorCamera,
    CreateMonoCamera,
    CreateImageManip,
    CreateVideoEncoder,

    CreateNeuralNetwork,
    CreateDetectionNetwork,
    CreateMobileNetDetectionNetwork,
    CreateMobileNetSpatialDetectionNetwork,
    CreateYoloDetectionNetwork,
    CreateYoloSpatialDetectionNetwork,
    CreateSpatialDetectionNetwork,

    CreateSPIIn,
    CreateXLinkIn,

    CreateSPIOut,
    CreateXLinkOut,

    CreateScript(i32, i32), // (n inputs, n outputs)

    CreateStereoDepth,
    CreateSpatialLocationCalculator,

    CreateEdgeDetector,
    CreateFeaureTracker,
    CreateObjectTracker,
    CreateIMU,
}

/// The response type is used to encode side-effects produced when drawing a
/// node in the graph. Most side-effects (creating new nodes, deleting existing
/// nodes, handling connections...) are already handled by the library, but this
/// mechanism allows creating additional side effects from user code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MyResponse {
    SetActiveNode(NodeId),
    ClearActiveNode,
}

/// The graph 'global' state. This state struct is passed around to the node and
/// parameter drawing callbacks. The contents of this struct are entirely up to
/// the user. For this example, we use it to keep track of the 'active' node.
#[derive(Default)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct MyGraphState {
    pub active_node: Option<NodeId>,
}

// =========== Then, you need to implement some traits ============

// A trait for the data types, to tell the library how to display them
impl DataTypeTrait<MyGraphState> for MyDataType {
    fn data_type_color(&self, _user_state: &mut MyGraphState) -> egui::Color32 {
        match self {
            MyDataType::Scalar => egui::Color32::from_rgb(38, 109, 211),
            MyDataType::Vec2 => egui::Color32::from_rgb(238, 207, 109),

            MyDataType::Queue(DepthaiNode::ColorCamera) => egui::Color32::from_rgb(241, 148, 138),
            MyDataType::Queue(DepthaiNode::MonoCamera) => egui::Color32::from_rgb(243, 243, 243),
            MyDataType::Queue(DepthaiNode::ImageManip) => egui::Color32::from_rgb(174, 214, 241),
            MyDataType::Queue(DepthaiNode::VideoEncoder) => egui::Color32::from_rgb(190, 190, 190),

            MyDataType::Queue(DepthaiNode::NeuralNetwork) => egui::Color32::from_rgb(171, 235, 198),
            MyDataType::Queue(DepthaiNode::DetectionNetwork) => {
                egui::Color32::from_rgb(171, 235, 198)
            }
            MyDataType::Queue(DepthaiNode::MobileNetDetectionNetwork) => {
                egui::Color32::from_rgb(171, 235, 198)
            }
            MyDataType::Queue(DepthaiNode::MobileNetSpatialDetectionNetwork) => {
                egui::Color32::from_rgb(171, 235, 198)
            }
            MyDataType::Queue(DepthaiNode::YoloDetectionNetwork) => {
                egui::Color32::from_rgb(171, 235, 198)
            }
            MyDataType::Queue(DepthaiNode::YoloSpatialDetectionNetwork) => {
                egui::Color32::from_rgb(171, 235, 198)
            }
            MyDataType::Queue(DepthaiNode::SpatialDetectionNetwork) => {
                egui::Color32::from_rgb(171, 235, 198)
            }

            MyDataType::Queue(DepthaiNode::SPIIn) => egui::Color32::from_rgb(242, 215, 213),
            MyDataType::Queue(DepthaiNode::XLinkIn) => egui::Color32::from_rgb(242, 215, 213),

            MyDataType::Queue(DepthaiNode::SPIOut) => egui::Color32::from_rgb(230, 176, 170),
            MyDataType::Queue(DepthaiNode::XLinkOut) => egui::Color32::from_rgb(230, 176, 170),

            MyDataType::Queue(DepthaiNode::Script) => egui::Color32::from_rgb(249, 231, 159),

            MyDataType::Queue(DepthaiNode::StereoDepth) => egui::Color32::from_rgb(215, 189, 226),
            MyDataType::Queue(DepthaiNode::SpatialLocationCalculator) => {
                egui::Color32::from_rgb(215, 189, 226)
            }

            MyDataType::Queue(DepthaiNode::EdgeDetector) => egui::Color32::from_rgb(248, 196, 113),
            MyDataType::Queue(DepthaiNode::FeaureTracker) => egui::Color32::from_rgb(248, 196, 113),
            MyDataType::Queue(DepthaiNode::ObjectTracker) => egui::Color32::from_rgb(248, 196, 113),
            MyDataType::Queue(DepthaiNode::IMU) => egui::Color32::from_rgb(248, 196, 113),
        }
    }

    fn name(&self) -> Cow<'_, str> {
        match self {
            MyDataType::Scalar => Cow::Borrowed("scalar"),
            MyDataType::Vec2 => Cow::Borrowed("2d vector"),
            MyDataType::Queue(DepthaiNode::ColorCamera) => Cow::Borrowed("Color Camera"),
            MyDataType::Queue(DepthaiNode::MonoCamera) => Cow::Borrowed("Mono Camera"),
            MyDataType::Queue(DepthaiNode::ImageManip) => Cow::Borrowed("Image Manipulation"),
            MyDataType::Queue(DepthaiNode::VideoEncoder) => Cow::Borrowed("Video Encoder"),

            MyDataType::Queue(DepthaiNode::NeuralNetwork) => Cow::Borrowed("Neural Network"),
            MyDataType::Queue(DepthaiNode::DetectionNetwork) => Cow::Borrowed("Detection Network"),
            MyDataType::Queue(DepthaiNode::MobileNetDetectionNetwork) => {
                Cow::Borrowed("MobileNet Detection Network")
            }
            MyDataType::Queue(DepthaiNode::MobileNetSpatialDetectionNetwork) => {
                Cow::Borrowed("MobileNet Spatial Detection Network")
            }
            MyDataType::Queue(DepthaiNode::YoloDetectionNetwork) => {
                Cow::Borrowed("Yolo Detection Network")
            }
            MyDataType::Queue(DepthaiNode::YoloSpatialDetectionNetwork) => {
                Cow::Borrowed("Yolo Spatial Detection Network")
            }
            MyDataType::Queue(DepthaiNode::SpatialDetectionNetwork) => {
                Cow::Borrowed("Spatial Detection Network")
            }

            MyDataType::Queue(DepthaiNode::SPIIn) => Cow::Borrowed("SPI In"),
            MyDataType::Queue(DepthaiNode::XLinkIn) => Cow::Borrowed("XLink In"),

            MyDataType::Queue(DepthaiNode::SPIOut) => Cow::Borrowed("SPI Out"),
            MyDataType::Queue(DepthaiNode::XLinkOut) => Cow::Borrowed("XLink Out"),

            MyDataType::Queue(DepthaiNode::Script) => Cow::Borrowed("Script"),

            MyDataType::Queue(DepthaiNode::StereoDepth) => Cow::Borrowed("Stereo Depth"),
            MyDataType::Queue(DepthaiNode::SpatialLocationCalculator) => {
                Cow::Borrowed("Spatial Location Calculator")
            }

            MyDataType::Queue(DepthaiNode::EdgeDetector) => Cow::Borrowed("Edge Detector"),
            MyDataType::Queue(DepthaiNode::FeaureTracker) => Cow::Borrowed("Feature Tracker"),
            MyDataType::Queue(DepthaiNode::ObjectTracker) => Cow::Borrowed("Object Tracker"),
            MyDataType::Queue(DepthaiNode::IMU) => Cow::Borrowed("IMU"),
        }
    }
}

// A trait for the node kinds, which tells the library how to build new nodes
// from the templates in the node finder
impl NodeTemplateTrait for MyNodeTemplate {
    type NodeData = MyNodeData;
    type DataType = MyDataType;
    type ValueType = MyValueType;
    type UserState = MyGraphState;
    type CategoryType = &'static str;

    fn node_finder_label(&self, _user_state: &mut Self::UserState) -> Cow<'_, str> {
        Cow::Borrowed(match self {
            MyNodeTemplate::MakeScalar => "New scalar",
            MyNodeTemplate::AddScalar => "Scalar add",
            MyNodeTemplate::SubtractScalar => "Scalar subtract",
            MyNodeTemplate::MakeVector => "New vector",
            MyNodeTemplate::AddVector => "Vector add",
            MyNodeTemplate::SubtractVector => "Vector subtract",
            MyNodeTemplate::VectorTimesScalar => "Vector times scalar",

            MyNodeTemplate::CreateColorCamera => "Create Color Camera",
            MyNodeTemplate::CreateMonoCamera => "Create Mono Camera",
            MyNodeTemplate::CreateImageManip => "Create Image Manipulation",
            MyNodeTemplate::CreateVideoEncoder => "Create Video Encoder",

            MyNodeTemplate::CreateNeuralNetwork => "Create Neural Network",
            MyNodeTemplate::CreateDetectionNetwork => "Create Detection Network",
            MyNodeTemplate::CreateMobileNetDetectionNetwork => "Create MobileNet Detection Network",
            MyNodeTemplate::CreateMobileNetSpatialDetectionNetwork => {
                "Create MobileNet Spatial Detection Network"
            }
            MyNodeTemplate::CreateYoloDetectionNetwork => "Create Yolo Detection Network",
            MyNodeTemplate::CreateYoloSpatialDetectionNetwork => {
                "Create Yolo Spatial Detection Network"
            }
            MyNodeTemplate::CreateSpatialDetectionNetwork => "Create Spatial Detection Network",

            MyNodeTemplate::CreateSPIIn => "Create SPI In",
            MyNodeTemplate::CreateXLinkIn => "Create XLink In",

            MyNodeTemplate::CreateSPIOut => "Create SPI Out",
            MyNodeTemplate::CreateXLinkOut => "Create XLink Out",

            MyNodeTemplate::CreateScript(_, _) => "Create Script",

            MyNodeTemplate::CreateStereoDepth => "Create Stereo Depth",
            MyNodeTemplate::CreateSpatialLocationCalculator => "Create Spatial Location Calculator",
            MyNodeTemplate::CreateEdgeDetector => "Create Edge Detector",
            MyNodeTemplate::CreateFeaureTracker => "Create Feature Tracker",
            MyNodeTemplate::CreateObjectTracker => "Create Object Tracker",
            MyNodeTemplate::CreateIMU => "Create IMU",
        })
    }

    // this is what allows the library to show collapsible lists in the node finder.
    fn node_finder_categories(&self, _user_state: &mut Self::UserState) -> Vec<&'static str> {
        match self {
            MyNodeTemplate::MakeScalar
            | MyNodeTemplate::AddScalar
            | MyNodeTemplate::SubtractScalar => vec!["Scalar"],
            MyNodeTemplate::MakeVector
            | MyNodeTemplate::AddVector
            | MyNodeTemplate::SubtractVector => vec!["Vector"],
            MyNodeTemplate::VectorTimesScalar => vec!["Vector", "Scalar"],
            _ => vec!["Other"],
        }
    }

    fn node_graph_label(&self, user_state: &mut Self::UserState) -> String {
        // It's okay to delegate this to node_finder_label if you don't want to
        // show different names in the node finder and the node itself.
        self.node_finder_label(user_state).into()
    }

    fn user_data(&self, _user_state: &mut Self::UserState) -> Self::NodeData {
        MyNodeData { template: *self }
    }

    fn build_node(
        &self,
        graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
        _user_state: &mut Self::UserState,
        node_id: NodeId,
    ) {
        let build_detection_network_node =
            |node_id: NodeId,
             graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
             depthai_node: DepthaiNode| {
                graph.add_input_param(
                    node_id,
                    "input".into(),
                    MyDataType::Queue(depthai_node),
                    MyValueType::Queue(depthai_node),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(node_id, "out".into(), MyDataType::Queue(depthai_node));
                graph.add_output_param(
                    node_id,
                    "outNetwork".into(),
                    MyDataType::Queue(depthai_node),
                );
                graph.add_output_param(
                    node_id,
                    "passthrough".into(),
                    MyDataType::Queue(depthai_node),
                );
            };

        let build_spatial_detection_network_node =
            |node_id: NodeId,
             graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
             depthai_node: DepthaiNode| {
                graph.add_input_param(
                    node_id,
                    "input".into(),
                    MyDataType::Queue(depthai_node),
                    MyValueType::Queue(depthai_node),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "inputDepth".into(),
                    MyDataType::Queue(depthai_node),
                    MyValueType::Queue(depthai_node),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(node_id, "out".into(), MyDataType::Queue(depthai_node));
                graph.add_output_param(
                    node_id,
                    "boundingBoxMapping".into(),
                    MyDataType::Queue(depthai_node),
                );
                graph.add_output_param(
                    node_id,
                    "passthroughDepth".into(),
                    MyDataType::Queue(depthai_node),
                );
                graph.add_output_param(
                    node_id,
                    "spatialLocationCalculatorOutput".into(),
                    MyDataType::Queue(depthai_node),
                );
                graph.add_output_param(
                    node_id,
                    "passthrough".into(),
                    MyDataType::Queue(depthai_node),
                );
            };

        // The nodes are created empty by default. This function needs to take
        // care of creating the desired inputs and outputs based on the template

        // We define some closures here to avoid boilerplate. Note that this is
        // entirely optional.
        let input_scalar = |graph: &mut MyGraph, name: &str| {
            graph.add_input_param(
                node_id,
                name.to_string(),
                MyDataType::Scalar,
                MyValueType::Scalar { value: 0.0 },
                InputParamKind::ConnectionOrConstant,
                true,
            );
        };
        let input_vector = |graph: &mut MyGraph, name: &str| {
            graph.add_input_param(
                node_id,
                name.to_string(),
                MyDataType::Vec2,
                MyValueType::Vec2 {
                    value: egui::vec2(0.0, 0.0),
                },
                InputParamKind::ConnectionOrConstant,
                true,
            );
        };

        let output_scalar = |graph: &mut MyGraph, name: &str| {
            graph.add_output_param(node_id, name.to_string(), MyDataType::Scalar);
        };
        let output_vector = |graph: &mut MyGraph, name: &str| {
            graph.add_output_param(node_id, name.to_string(), MyDataType::Vec2);
        };

        match self {
            MyNodeTemplate::AddScalar => {
                // The first input param doesn't use the closure so we can comment
                // it in more detail.
                graph.add_input_param(
                    node_id,
                    // This is the name of the parameter. Can be later used to
                    // retrieve the value. Parameter names should be unique.
                    "A".into(),
                    // The data type for this input. In this case, a scalar
                    MyDataType::Scalar,
                    // The value type for this input. We store zero as default
                    MyValueType::Scalar { value: 0.0 },
                    // The input parameter kind. This allows defining whether a
                    // parameter accepts input connections and/or an inline
                    // widget to set its value.
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                input_scalar(graph, "B");
                output_scalar(graph, "out");
            }
            MyNodeTemplate::SubtractScalar => {
                input_scalar(graph, "A");
                input_scalar(graph, "B");
                output_scalar(graph, "out");
            }
            MyNodeTemplate::VectorTimesScalar => {
                input_scalar(graph, "scalar");
                input_vector(graph, "vector");
                output_vector(graph, "out");
            }
            MyNodeTemplate::AddVector => {
                input_vector(graph, "v1");
                input_vector(graph, "v2");
                output_vector(graph, "out");
            }
            MyNodeTemplate::SubtractVector => {
                input_vector(graph, "v1");
                input_vector(graph, "v2");
                output_vector(graph, "out");
            }
            MyNodeTemplate::MakeVector => {
                input_scalar(graph, "x");
                input_scalar(graph, "y");
                output_vector(graph, "out");
            }
            MyNodeTemplate::MakeScalar => {
                input_scalar(graph, "value");
                output_scalar(graph, "out");
            }

            MyNodeTemplate::CreateColorCamera => {
                graph.add_input_param(
                    node_id,
                    "inputConfig".into(),
                    MyDataType::Queue(DepthaiNode::ColorCamera),
                    MyValueType::Queue(DepthaiNode::ColorCamera),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "inputControl".into(),
                    MyDataType::Queue(DepthaiNode::ColorCamera),
                    MyValueType::Queue(DepthaiNode::ColorCamera),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "raw".into(),
                    MyDataType::Queue(DepthaiNode::ColorCamera),
                );
                graph.add_output_param(
                    node_id,
                    "isp".into(),
                    MyDataType::Queue(DepthaiNode::ColorCamera),
                );
                graph.add_output_param(
                    node_id,
                    "video".into(),
                    MyDataType::Queue(DepthaiNode::ColorCamera),
                );
                graph.add_output_param(
                    node_id,
                    "still".into(),
                    MyDataType::Queue(DepthaiNode::ColorCamera),
                );
                graph.add_output_param(
                    node_id,
                    "preview".into(),
                    MyDataType::Queue(DepthaiNode::ColorCamera),
                );
            }
            MyNodeTemplate::CreateMonoCamera => {
                graph.add_input_param(
                    node_id,
                    "inputControl".into(),
                    MyDataType::Queue(DepthaiNode::MonoCamera),
                    MyValueType::Queue(DepthaiNode::MonoCamera),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "out".into(),
                    MyDataType::Queue(DepthaiNode::MonoCamera {}),
                );
                graph.add_output_param(
                    node_id,
                    "raw".into(),
                    MyDataType::Queue(DepthaiNode::MonoCamera {}),
                );
            }

            MyNodeTemplate::CreateImageManip => {
                graph.add_input_param(
                    node_id,
                    "inputImage".into(),
                    MyDataType::Queue(DepthaiNode::ImageManip),
                    MyValueType::Queue(DepthaiNode::ImageManip),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "inputConfig".into(),
                    MyDataType::Queue(DepthaiNode::ImageManip),
                    MyValueType::Queue(DepthaiNode::ImageManip),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "out".into(),
                    MyDataType::Queue(DepthaiNode::ImageManip {}),
                );
            }
            MyNodeTemplate::CreateVideoEncoder => {
                graph.add_input_param(
                    node_id,
                    "input".into(),
                    MyDataType::Queue(DepthaiNode::VideoEncoder),
                    MyValueType::Queue(DepthaiNode::VideoEncoder),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "bitstream".into(),
                    MyDataType::Queue(DepthaiNode::VideoEncoder {}),
                );
            }
            MyNodeTemplate::CreateNeuralNetwork => {
                graph.add_input_param(
                    node_id,
                    "input".into(),
                    MyDataType::Queue(DepthaiNode::NeuralNetwork),
                    MyValueType::Queue(DepthaiNode::NeuralNetwork),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "out".into(),
                    MyDataType::Queue(DepthaiNode::NeuralNetwork {}),
                );
                graph.add_output_param(
                    node_id,
                    "passthrough".into(),
                    MyDataType::Queue(DepthaiNode::NeuralNetwork {}),
                );
            }

            MyNodeTemplate::CreateDetectionNetwork => {
                build_detection_network_node(node_id, graph, DepthaiNode::DetectionNetwork {});
            }
            MyNodeTemplate::CreateMobileNetDetectionNetwork => {
                build_detection_network_node(
                    node_id,
                    graph,
                    DepthaiNode::MobileNetDetectionNetwork {},
                );
            }
            MyNodeTemplate::CreateMobileNetSpatialDetectionNetwork => {
                build_spatial_detection_network_node(
                    node_id,
                    graph,
                    DepthaiNode::MobileNetSpatialDetectionNetwork {},
                );
            }
            MyNodeTemplate::CreateYoloDetectionNetwork => {
                build_detection_network_node(node_id, graph, DepthaiNode::YoloDetectionNetwork {});
            }
            MyNodeTemplate::CreateYoloSpatialDetectionNetwork => {
                build_spatial_detection_network_node(
                    node_id,
                    graph,
                    DepthaiNode::YoloSpatialDetectionNetwork {},
                );
            }
            MyNodeTemplate::CreateSPIIn => {
                graph.add_input_param(
                    node_id,
                    "SPI (from MCU)".into(),
                    MyDataType::Queue(DepthaiNode::SPIIn),
                    MyValueType::Queue(DepthaiNode::SPIIn),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "out".into(),
                    MyDataType::Queue(DepthaiNode::SPIIn),
                );
            }

            MyNodeTemplate::CreateSPIOut => {
                graph.add_input_param(
                    node_id,
                    "input".into(),
                    MyDataType::Queue(DepthaiNode::SPIOut),
                    MyValueType::Queue(DepthaiNode::SPIOut),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "SPI (to MCU)".into(),
                    MyDataType::Queue(DepthaiNode::SPIOut),
                );
            }

            MyNodeTemplate::CreateXLinkOut => {
                graph.add_input_param(
                    node_id,
                    "input".into(),
                    MyDataType::Queue(DepthaiNode::XLinkOut),
                    MyValueType::Queue(DepthaiNode::XLinkOut),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );

                graph.add_output_param(
                    node_id,
                    "(to host)".into(),
                    MyDataType::Queue(DepthaiNode::XLinkOut {}),
                );
            }
            MyNodeTemplate::CreateXLinkIn => {
                graph.add_input_param(
                    node_id,
                    "(from host)".into(),
                    MyDataType::Queue(DepthaiNode::XLinkIn),
                    MyValueType::Queue(DepthaiNode::XLinkIn),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );

                graph.add_output_param(
                    node_id,
                    "out".into(),
                    MyDataType::Queue(DepthaiNode::XLinkIn {}),
                );
            }

            MyNodeTemplate::CreateScript(n_inputs, n_outputs) => {
                for inp in 0..*n_inputs {
                    graph.add_input_param(
                        node_id,
                        format!("in{}", inp).into(),
                        MyDataType::Queue(DepthaiNode::Script),
                        MyValueType::Queue(DepthaiNode::Script),
                        InputParamKind::ConnectionOrConstant,
                        true,
                    );
                }

                for out in 0..*n_outputs {
                    graph.add_output_param(
                        node_id,
                        format!("out{}", out).into(),
                        MyDataType::Queue(DepthaiNode::Script {}),
                    );
                }
            }
            MyNodeTemplate::CreateStereoDepth => {
                graph.add_input_param(
                    node_id,
                    "inputConfig".into(),
                    MyDataType::Queue(DepthaiNode::StereoDepth),
                    MyValueType::Queue(DepthaiNode::StereoDepth),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "left".into(),
                    MyDataType::Queue(DepthaiNode::StereoDepth),
                    MyValueType::Queue(DepthaiNode::StereoDepth),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "right".into(),
                    MyDataType::Queue(DepthaiNode::StereoDepth),
                    MyValueType::Queue(DepthaiNode::StereoDepth),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );

                graph.add_output_param(
                    node_id,
                    "depth".into(),
                    MyDataType::Queue(DepthaiNode::StereoDepth {}),
                );
                graph.add_output_param(
                    node_id,
                    "disparity".into(),
                    MyDataType::Queue(DepthaiNode::StereoDepth {}),
                );
                // graph.add_output_param(
                //     node_id,
                //     "rectifiedLeft".into(),
                //     MyDataType::Queue(DepthaiNode::StereoDepth {}),
                // );
                // graph.add_output_param(
                //     node_id,
                //     "rectifiedRight".into(),
                //     MyDataType::Queue(DepthaiNode::StereoDepth {}),
                // );
                // graph.add_output_param(
                //     node_id,
                //     "syncedLeft".into(),
                //     MyDataType::Queue(DepthaiNode::StereoDepth {}),
                // );
                // graph.add_output_param(
                //     node_id,
                //     "syncedRight".into(),
                //     MyDataType::Queue(DepthaiNode::StereoDepth {}),
                // );
                // graph.add_output_param(
                //     node_id,
                //     "confidenceMap".into(),
                //     MyDataType::Queue(DepthaiNode::StereoDepth {}),
                // );
                // TODO(filip): Add more outputs
            }
            MyNodeTemplate::CreateSpatialLocationCalculator => {
                graph.add_input_param(
                    node_id,
                    "inputConfig".into(),
                    MyDataType::Queue(DepthaiNode::SpatialLocationCalculator),
                    MyValueType::Queue(DepthaiNode::SpatialLocationCalculator),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "inputDepth".into(),
                    MyDataType::Queue(DepthaiNode::SpatialLocationCalculator),
                    MyValueType::Queue(DepthaiNode::SpatialLocationCalculator),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "out".into(),
                    MyDataType::Queue(DepthaiNode::SpatialLocationCalculator {}),
                );
                graph.add_output_param(
                    node_id,
                    "passthroughDepth".into(),
                    MyDataType::Queue(DepthaiNode::SpatialLocationCalculator {}),
                );
            }
            MyNodeTemplate::CreateEdgeDetector => {
                graph.add_input_param(
                    node_id,
                    "inputConfig".into(),
                    MyDataType::Queue(DepthaiNode::EdgeDetector),
                    MyValueType::Queue(DepthaiNode::EdgeDetector),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "inputImage".into(),
                    MyDataType::Queue(DepthaiNode::EdgeDetector),
                    MyValueType::Queue(DepthaiNode::EdgeDetector),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "outputImage".into(),
                    MyDataType::Queue(DepthaiNode::EdgeDetector {}),
                );
            }
            MyNodeTemplate::CreateFeaureTracker => {
                graph.add_input_param(
                    node_id,
                    "inputConfig".into(),
                    MyDataType::Queue(DepthaiNode::FeaureTracker),
                    MyValueType::Queue(DepthaiNode::FeaureTracker),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "inputImage".into(),
                    MyDataType::Queue(DepthaiNode::FeaureTracker),
                    MyValueType::Queue(DepthaiNode::FeaureTracker),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "outputFeatures".into(),
                    MyDataType::Queue(DepthaiNode::FeaureTracker {}),
                );
                graph.add_output_param(
                    node_id,
                    "passthroughInputImage".into(),
                    MyDataType::Queue(DepthaiNode::FeaureTracker {}),
                );
            }
            MyNodeTemplate::CreateObjectTracker => {
                graph.add_input_param(
                    node_id,
                    "inputDetectionFrame".into(),
                    MyDataType::Queue(DepthaiNode::ObjectTracker),
                    MyValueType::Queue(DepthaiNode::ObjectTracker),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "inputDetections".into(),
                    MyDataType::Queue(DepthaiNode::ObjectTracker),
                    MyValueType::Queue(DepthaiNode::ObjectTracker),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "inputTrackerFrame".into(),
                    MyDataType::Queue(DepthaiNode::ObjectTracker),
                    MyValueType::Queue(DepthaiNode::ObjectTracker),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
                graph.add_output_param(
                    node_id,
                    "out".into(),
                    MyDataType::Queue(DepthaiNode::ObjectTracker {}),
                );
                graph.add_output_param(
                    node_id,
                    "passthroughDetectionFrame".into(),
                    MyDataType::Queue(DepthaiNode::ObjectTracker {}),
                );
                graph.add_output_param(
                    node_id,
                    "passthroughDetections".into(),
                    MyDataType::Queue(DepthaiNode::ObjectTracker {}),
                );
                graph.add_output_param(
                    node_id,
                    "passthroughTrackerFrame".into(),
                    MyDataType::Queue(DepthaiNode::ObjectTracker {}),
                );
            }
            MyNodeTemplate::CreateIMU => {
                graph.add_output_param(
                    node_id,
                    "out".into(),
                    MyDataType::Queue(DepthaiNode::IMU {}),
                );
            }
            MyNodeTemplate::CreateSpatialDetectionNetwork => {
                build_spatial_detection_network_node(
                    node_id,
                    graph,
                    DepthaiNode::SpatialDetectionNetwork {},
                );
            }
        }
    }
}

pub struct AllMyNodeTemplates;
impl NodeTemplateIter for AllMyNodeTemplates {
    type Item = MyNodeTemplate;

    fn all_kinds(&self) -> Vec<Self::Item> {
        // This function must return a list of node kinds, which the node finder
        // will use to display it to the user. Crates like strum can reduce the
        // boilerplate in enumerating all variants of an enum.
        vec![
            MyNodeTemplate::MakeScalar,
            MyNodeTemplate::MakeVector,
            MyNodeTemplate::AddScalar,
            MyNodeTemplate::SubtractScalar,
            MyNodeTemplate::AddVector,
            MyNodeTemplate::SubtractVector,
            MyNodeTemplate::VectorTimesScalar,
        ]
    }
}

impl WidgetValueTrait for MyValueType {
    type Response = MyResponse;
    type UserState = MyGraphState;
    type NodeData = MyNodeData;
    fn value_widget(
        &mut self,
        param_name: &str,
        _node_id: NodeId,
        ui: &mut egui::Ui,
        _user_state: &mut MyGraphState,
        _node_data: &MyNodeData,
    ) -> Vec<MyResponse> {
        // This trait is used to tell the library which UI to display for the
        // inline parameter widgets.
        match self {
            MyValueType::Vec2 { value } => {
                ui.label(param_name);
                ui.horizontal(|ui| {
                    ui.label("x");
                    ui.add(DragValue::new(&mut value.x));
                    ui.label("y");
                    ui.add(DragValue::new(&mut value.y));
                });
            }
            MyValueType::Scalar { value } => {
                ui.horizontal(|ui| {
                    ui.label(param_name);
                    ui.add(DragValue::new(value));
                });
            }
            _ => {
                ui.label(param_name);
            }
        }
        // This allows you to return your responses from the inline widgets.
        Vec::new()
    }
}

impl UserResponseTrait for MyResponse {}
impl NodeDataTrait for MyNodeData {
    type Response = MyResponse;
    type UserState = MyGraphState;
    type DataType = MyDataType;
    type ValueType = MyValueType;

    // This method will be called when drawing each node. This allows adding
    // extra ui elements inside the nodes. In this case, we create an "active"
    // button which introduces the concept of having an active node in the
    // graph. This is done entirely from user code with no modifications to the
    // node graph library.
    fn bottom_ui(
        &self,
        ui: &mut egui::Ui,
        node_id: NodeId,
        _graph: &Graph<MyNodeData, MyDataType, MyValueType>,
        user_state: &mut Self::UserState,
    ) -> Vec<NodeResponse<MyResponse, MyNodeData>>
    where
        MyResponse: UserResponseTrait,
    {
        // This logic is entirely up to the user. In this case, we check if the
        // current node we're drawing is the active one, by comparing against
        // the value stored in the global user state, and draw different button
        // UIs based on that.

        let mut responses = vec![];
        let is_active = user_state
            .active_node
            .map(|id| id == node_id)
            .unwrap_or(false);

        // Pressing the button will emit a custom user response to either set,
        // or clear the active node. These responses do nothing by themselves,
        // the library only makes the responses available to you after the graph
        // has been drawn. See below at the update method for an example.
        if !is_active {
            if ui.button("👁 Set active").clicked() {
                responses.push(NodeResponse::User(MyResponse::SetActiveNode(node_id)));
            }
        } else {
            let button =
                egui::Button::new(egui::RichText::new("👁 Active").color(egui::Color32::BLACK))
                    .fill(egui::Color32::GOLD);
            if ui.add(button).clicked() {
                responses.push(NodeResponse::User(MyResponse::ClearActiveNode));
            }
        }

        responses
    }

    fn bg_and_text_color(
        &self,
        _ui: &egui::Ui,
        _node_id: NodeId,
        _graph: &Graph<Self, Self::DataType, Self::ValueType>,
        _user_state: &mut Self::UserState,
    ) -> Option<(egui::Color32, egui::Color32)> {
        match _graph.nodes[_node_id].user_data.template {
            MyNodeTemplate::CreateColorCamera => Some((
                egui::Color32::from_rgb(241, 148, 138),
                egui::Color32::from_rgb(0, 0, 0),
            )),
            MyNodeTemplate::CreateMonoCamera => Some((
                egui::Color32::from_rgb(243, 243, 243),
                egui::Color32::from_rgb(0, 0, 0),
            )),
            MyNodeTemplate::CreateImageManip => Some((
                egui::Color32::from_rgb(174, 214, 241),
                egui::Color32::from_rgb(0, 0, 0),
            )),
            MyNodeTemplate::CreateVideoEncoder => Some((
                egui::Color32::from_rgb(190, 190, 190),
                egui::Color32::from_rgb(0, 0, 0),
            )),
            MyNodeTemplate::CreateNeuralNetwork
            | MyNodeTemplate::CreateDetectionNetwork
            | MyNodeTemplate::CreateMobileNetDetectionNetwork
            | MyNodeTemplate::CreateMobileNetSpatialDetectionNetwork
            | MyNodeTemplate::CreateYoloDetectionNetwork
            | MyNodeTemplate::CreateYoloSpatialDetectionNetwork => Some((
                egui::Color32::from_rgb(171, 235, 198),
                egui::Color32::from_rgb(0, 0, 0),
            )),
            MyNodeTemplate::CreateSPIIn | MyNodeTemplate::CreateXLinkIn => Some((
                egui::Color32::from_rgb(242, 215, 213),
                egui::Color32::from_rgb(0, 0, 0),
            )),
            MyNodeTemplate::CreateSPIOut | MyNodeTemplate::CreateXLinkOut => Some((
                egui::Color32::from_rgb(230, 176, 170),
                egui::Color32::from_rgb(0, 0, 0),
            )),
            MyNodeTemplate::CreateScript(..) => Some((
                egui::Color32::from_rgb(249, 231, 159),
                egui::Color32::from_rgb(0, 0, 0),
            )),
            MyNodeTemplate::CreateStereoDepth | MyNodeTemplate::CreateSpatialLocationCalculator => {
                Some((
                    egui::Color32::from_rgb(215, 189, 226),
                    egui::Color32::from_rgb(0, 0, 0),
                ))
            }
            MyNodeTemplate::CreateEdgeDetector
            | MyNodeTemplate::CreateFeaureTracker
            | MyNodeTemplate::CreateObjectTracker
            | MyNodeTemplate::CreateIMU => Some((
                egui::Color32::from_rgb(248, 196, 113),
                egui::Color32::from_rgb(0, 0, 0),
            )),

            _ => None,
        }
    }
}

type MyGraph = Graph<MyNodeData, MyDataType, MyValueType>;
type MyEditorState =
    GraphEditorState<MyNodeData, MyDataType, MyValueType, MyNodeTemplate, MyGraphState>;

#[derive(Default)]
pub struct NodeGraphExample {
    // The `GraphEditorState` is the top-level object. You "register" all your
    // custom types by specifying it as its generic parameters.
    state: MyEditorState,

    user_state: MyGraphState,
}

#[cfg(feature = "persistence")]
const PERSISTENCE_KEY: &str = "egui_node_graph";

#[cfg(feature = "persistence")]
impl NodeGraphExample {
    /// If the persistence feature is enabled, Called once before the first frame.
    /// Load previous app state (if any).
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let state = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, PERSISTENCE_KEY))
            .unwrap_or_default();
        Self {
            state,
            user_state: MyGraphState::default(),
        }
    }
}

impl eframe::App for NodeGraphExample {
    #[cfg(feature = "persistence")]
    /// If the persistence function is enabled,
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, PERSISTENCE_KEY, &self.state);
    }
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                egui::widgets::global_dark_light_mode_switch(ui);
            });
        });
        let graph_response = egui::CentralPanel::default()
            .show(ctx, |ui| {
                self.state.draw_graph_editor(
                    ui,
                    AllMyNodeTemplates,
                    &mut self.user_state,
                    Vec::default(),
                )
            })
            .inner;
        for node_response in graph_response.node_responses {
            // Here, we ignore all other graph events. But you may find
            // some use for them. For example, by playing a sound when a new
            // connection is created
            if let NodeResponse::User(user_event) = node_response {
                match user_event {
                    MyResponse::SetActiveNode(node) => self.user_state.active_node = Some(node),
                    MyResponse::ClearActiveNode => self.user_state.active_node = None,
                }
            }
        }

        // No nodes, create from json
        if self.state.node_positions.is_empty() {
            // Create color cam and xlink out

            let color_cam_template = MyNodeTemplate::CreateColorCamera {};
            let color_cam = self.state.graph.add_node(
                "Color Camera".to_owned(),
                MyNodeData {
                    template: color_cam_template,
                },
                |g, node_id| color_cam_template.build_node(g, &mut self.user_state, node_id),
            );
            self.state
                .node_positions
                .insert(color_cam, egui::emath::Pos2::new(0.0, 0.0));
            self.state.node_order.push(color_cam);

            let xlinnk_out_template = MyNodeTemplate::CreateXLinkOut {};
            let xlink_out = self.state.graph.add_node(
                "Xlink Out".to_owned(),
                MyNodeData {
                    template: xlinnk_out_template,
                },
                |g, node_id| xlinnk_out_template.build_node(g, &mut self.user_state, node_id),
            );
            self.state
                .node_positions
                .insert(xlink_out, egui::emath::Pos2::new(0.0, 1.0));
            self.state.node_order.push(xlink_out);
        }

        if let Some(node) = self.user_state.active_node {
            if self.state.graph.nodes.contains_key(node) {
                let text = match evaluate_node(&self.state.graph, node, &mut HashMap::new()) {
                    Ok(value) => format!("The result is: {:?}", value),
                    Err(err) => format!("Execution error: {}", err),
                };
                ctx.debug_painter().text(
                    egui::pos2(10.0, 35.0),
                    egui::Align2::LEFT_TOP,
                    text,
                    TextStyle::Button.resolve(&ctx.style()),
                    egui::Color32::WHITE,
                );
            } else {
                self.user_state.active_node = None;
            }
        }
    }
}

type OutputsCache = HashMap<OutputId, MyValueType>;

/// Recursively evaluates all dependencies of this node, then evaluates the node itself.
pub fn evaluate_node(
    graph: &MyGraph,
    node_id: NodeId,
    outputs_cache: &mut OutputsCache,
) -> anyhow::Result<MyValueType> {
    // To solve a similar problem as creating node types above, we define an
    // Evaluator as a convenience. It may be overkill for this small example,
    // but something like this makes the code much more readable when the
    // number of nodes starts growing.

    struct Evaluator<'a> {
        graph: &'a MyGraph,
        outputs_cache: &'a mut OutputsCache,
        node_id: NodeId,
    }
    impl<'a> Evaluator<'a> {
        fn new(graph: &'a MyGraph, outputs_cache: &'a mut OutputsCache, node_id: NodeId) -> Self {
            Self {
                graph,
                outputs_cache,
                node_id,
            }
        }
        fn evaluate_input(&mut self, name: &str) -> anyhow::Result<MyValueType> {
            // Calling `evaluate_input` recursively evaluates other nodes in the
            // graph until the input value for a paramater has been computed.
            evaluate_input(self.graph, self.node_id, name, self.outputs_cache)
        }
        fn populate_output(
            &mut self,
            name: &str,
            value: MyValueType,
        ) -> anyhow::Result<MyValueType> {
            // After computing an output, we don't just return it, but we also
            // populate the outputs cache with it. This ensures the evaluation
            // only ever computes an output once.
            //
            // The return value of the function is the "final" output of the
            // node, the thing we want to get from the evaluation. The example
            // would be slightly more contrived when we had multiple output
            // values, as we would need to choose which of the outputs is the
            // one we want to return. Other outputs could be used as
            // intermediate values.
            //
            // Note that this is just one possible semantic interpretation of
            // the graphs, you can come up with your own evaluation semantics!
            populate_output(self.graph, self.outputs_cache, self.node_id, name, value)
        }
        fn input_vector(&mut self, name: &str) -> anyhow::Result<egui::Vec2> {
            self.evaluate_input(name)?.try_to_vec2()
        }
        fn input_scalar(&mut self, name: &str) -> anyhow::Result<f32> {
            self.evaluate_input(name)?.try_to_scalar()
        }
        fn output_vector(&mut self, name: &str, value: egui::Vec2) -> anyhow::Result<MyValueType> {
            self.populate_output(name, MyValueType::Vec2 { value })
        }
        fn output_scalar(&mut self, name: &str, value: f32) -> anyhow::Result<MyValueType> {
            self.populate_output(name, MyValueType::Scalar { value })
        }
        fn input_queue(&mut self, name: &str) -> anyhow::Result<DepthaiNode> {
            self.evaluate_input(name)?.try_to_node()
        }
        fn output_queue(&mut self, name: &str, value: DepthaiNode) -> anyhow::Result<MyValueType> {
            self.populate_output(name, MyValueType::Queue(value))
        }
    }

    let node = &graph[node_id];
    let mut evaluator = Evaluator::new(graph, outputs_cache, node_id);
    match node.user_data.template {
        MyNodeTemplate::AddScalar => {
            let a = evaluator.input_scalar("A")?;
            let b = evaluator.input_scalar("B")?;
            evaluator.output_scalar("out", a + b)
        }
        MyNodeTemplate::SubtractScalar => {
            let a = evaluator.input_scalar("A")?;
            let b = evaluator.input_scalar("B")?;
            evaluator.output_scalar("out", a - b)
        }
        MyNodeTemplate::VectorTimesScalar => {
            let scalar = evaluator.input_scalar("scalar")?;
            let vector = evaluator.input_vector("vector")?;
            evaluator.output_vector("out", vector * scalar)
        }
        MyNodeTemplate::AddVector => {
            let v1 = evaluator.input_vector("v1")?;
            let v2 = evaluator.input_vector("v2")?;
            evaluator.output_vector("out", v1 + v2)
        }
        MyNodeTemplate::SubtractVector => {
            let v1 = evaluator.input_vector("v1")?;
            let v2 = evaluator.input_vector("v2")?;
            evaluator.output_vector("out", v1 - v2)
        }
        MyNodeTemplate::MakeVector => {
            let x = evaluator.input_scalar("x")?;
            let y = evaluator.input_scalar("y")?;
            evaluator.output_vector("out", egui::vec2(x, y))
        }
        MyNodeTemplate::MakeScalar => {
            let value = evaluator.input_scalar("value")?;
            evaluator.output_scalar("out", value)
        }
        MyNodeTemplate::CreateColorCamera => {
            let input_config = evaluator.input_queue("inputConfig")?;
            evaluator.output_queue("video", input_config)
        }
        MyNodeTemplate::CreateXLinkOut => {
            let input = evaluator.input_queue("input")?;
            evaluator.output_queue("output", input)
        }
        _ => {
            let value = evaluator.input_scalar("value")?;
            evaluator.output_scalar("out", value)
        }
    }
}

fn populate_output(
    graph: &MyGraph,
    outputs_cache: &mut OutputsCache,
    node_id: NodeId,
    param_name: &str,
    value: MyValueType,
) -> anyhow::Result<MyValueType> {
    let output_id = graph[node_id].get_output(param_name)?;
    outputs_cache.insert(output_id, value);
    Ok(value)
}

// Evaluates the input value of
fn evaluate_input(
    graph: &MyGraph,
    node_id: NodeId,
    param_name: &str,
    outputs_cache: &mut OutputsCache,
) -> anyhow::Result<MyValueType> {
    let input_id = graph[node_id].get_input(param_name)?;

    // The output of another node is connected.
    if let Some(other_output_id) = graph.connection(input_id) {
        // The value was already computed due to the evaluation of some other
        // node. We simply return value from the cache.
        if let Some(other_value) = outputs_cache.get(&other_output_id) {
            Ok(*other_value)
        }
        // This is the first time encountering this node, so we need to
        // recursively evaluate it.
        else {
            // Calling this will populate the cache
            evaluate_node(graph, graph[other_output_id].node, outputs_cache)?;

            // Now that we know the value is cached, return it
            Ok(*outputs_cache
                .get(&other_output_id)
                .expect("Cache should be populated"))
        }
    }
    // No existing connection, take the inline value instead.
    else {
        Ok(graph[input_id].value)
    }
}
