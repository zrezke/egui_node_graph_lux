use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    hash::Hash,
};

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
#[derive(Clone)]
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

    CreateScript(Vec<String>, Vec<String>), // (input names, output names)

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
        MyNodeData {
            template: self.clone(),
        }
    }

    fn build_node(
        &self,
        graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
        _user_state: &mut Self::UserState,
        node_id: NodeId,
    ) {
        let build_depthai_input =
            |node_id: NodeId,
             graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
             name: String,
             depthai_node: DepthaiNode| {
                graph.add_input_param(
                    node_id,
                    name,
                    MyDataType::Queue(depthai_node),
                    MyValueType::Queue(depthai_node),
                    InputParamKind::ConnectionOrConstant,
                    true,
                );
            };
        let build_depthai_output =
            |node_id: NodeId,
             graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
             name: String,
             depthai_node: DepthaiNode| {
                graph.add_output_param(node_id, name, MyDataType::Queue(depthai_node));
            };

        let build_detection_network_node =
            |node_id: NodeId,
             graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
             depthai_node: DepthaiNode| {
                build_depthai_input(node_id, graph, "in".into(), depthai_node);
                build_depthai_output(node_id, graph, "out".into(), depthai_node);
                graph.add_output_param(
                    node_id,
                    "passthrough".into(),
                    MyDataType::Queue(depthai_node),
                );
                build_depthai_output(node_id, graph, "outNetwork".into(), depthai_node);
            };

        let build_spatial_detection_network_node =
            |node_id: NodeId,
             graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
             depthai_node: DepthaiNode| {
                build_depthai_input(node_id, graph, "in".into(), depthai_node);
                build_depthai_input(node_id, graph, "inputDepth".into(), depthai_node);
                build_depthai_output(node_id, graph, "out".into(), depthai_node);
                build_depthai_output(node_id, graph, "boundingBoxMapping".into(), depthai_node);
                build_depthai_output(node_id, graph, "passthroughDepth".into(), depthai_node);
                build_depthai_output(
                    node_id,
                    graph,
                    "spatialLocationCalculatorOutput".into(),
                    depthai_node,
                );
                build_depthai_output(node_id, graph, "passthrough".into(), depthai_node);
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
                build_depthai_input(
                    node_id,
                    graph,
                    "inputConfig".into(),
                    DepthaiNode::ColorCamera,
                );
                build_depthai_input(
                    node_id,
                    graph,
                    "inputControl".into(),
                    DepthaiNode::ColorCamera,
                );
                build_depthai_output(node_id, graph, "raw".into(), DepthaiNode::ColorCamera);
                build_depthai_output(node_id, graph, "isp".into(), DepthaiNode::ColorCamera);
                build_depthai_output(node_id, graph, "video".into(), DepthaiNode::ColorCamera);
                build_depthai_output(node_id, graph, "still".into(), DepthaiNode::ColorCamera);
                build_depthai_output(node_id, graph, "preview".into(), DepthaiNode::ColorCamera);
            }
            MyNodeTemplate::CreateMonoCamera => {
                build_depthai_input(
                    node_id,
                    graph,
                    "inputConfig".into(),
                    DepthaiNode::MonoCamera,
                );
                build_depthai_input(
                    node_id,
                    graph,
                    "inputControl".into(),
                    DepthaiNode::MonoCamera,
                );
                build_depthai_output(node_id, graph, "out".into(), DepthaiNode::MonoCamera);
            }

            MyNodeTemplate::CreateImageManip => {
                build_depthai_input(node_id, graph, "inputImage".into(), DepthaiNode::ImageManip);
                build_depthai_input(
                    node_id,
                    graph,
                    "inputConfig".into(),
                    DepthaiNode::ImageManip,
                );
                build_depthai_output(node_id, graph, "out".into(), DepthaiNode::ImageManip);
            }
            MyNodeTemplate::CreateVideoEncoder => {
                build_depthai_input(node_id, graph, "in".into(), DepthaiNode::VideoEncoder);
                build_depthai_output(
                    node_id,
                    graph,
                    "bitstream".into(),
                    DepthaiNode::VideoEncoder,
                );
            }
            MyNodeTemplate::CreateNeuralNetwork => {
                build_depthai_input(node_id, graph, "in".into(), DepthaiNode::NeuralNetwork);
                build_depthai_output(node_id, graph, "out".into(), DepthaiNode::NeuralNetwork);
                build_depthai_output(
                    node_id,
                    graph,
                    "passthrough".into(),
                    DepthaiNode::NeuralNetwork,
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
                build_depthai_input(node_id, graph, "SPI (from MCU)".into(), DepthaiNode::SPIIn);
                build_depthai_output(node_id, graph, "out".into(), DepthaiNode::SPIIn);
            }

            MyNodeTemplate::CreateSPIOut => {
                build_depthai_input(node_id, graph, "in".into(), DepthaiNode::SPIOut);
                build_depthai_output(node_id, graph, "SPI (to MCU)".into(), DepthaiNode::SPIOut);
            }

            MyNodeTemplate::CreateXLinkOut => {
                build_depthai_input(node_id, graph, "in".into(), DepthaiNode::XLinkOut);
                build_depthai_output(
                    node_id,
                    graph,
                    "XLink (to host)".into(),
                    DepthaiNode::XLinkOut,
                );
            }
            MyNodeTemplate::CreateXLinkIn => {
                build_depthai_input(
                    node_id,
                    graph,
                    "XLink (from host)".into(),
                    DepthaiNode::XLinkIn,
                );
                build_depthai_output(node_id, graph, "out".into(), DepthaiNode::XLinkIn);
            }

            MyNodeTemplate::CreateScript(input_names, output_names) => {
                for inp in input_names {
                    build_depthai_input(node_id, graph, inp.into(), DepthaiNode::Script);
                }

                for out in output_names {
                    build_depthai_output(node_id, graph, out.into(), DepthaiNode::Script);
                }
            }
            MyNodeTemplate::CreateStereoDepth => {
                build_depthai_input(
                    node_id,
                    graph,
                    "inputConfig".into(),
                    DepthaiNode::StereoDepth,
                );
                build_depthai_input(node_id, graph, "left".into(), DepthaiNode::StereoDepth);
                build_depthai_input(node_id, graph, "right".into(), DepthaiNode::StereoDepth);
                build_depthai_output(node_id, graph, "depth".into(), DepthaiNode::StereoDepth);
                build_depthai_output(node_id, graph, "disparity".into(), DepthaiNode::StereoDepth);
                build_depthai_output(
                    node_id,
                    graph,
                    "rectifiedLeft".into(),
                    DepthaiNode::StereoDepth,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "rectifiedRight".into(),
                    DepthaiNode::StereoDepth,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "syncedLeft".into(),
                    DepthaiNode::StereoDepth,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "syncedRight".into(),
                    DepthaiNode::StereoDepth,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "confidenceMap".into(),
                    DepthaiNode::StereoDepth,
                );
                // TODO(filip): Add more outputs
            }
            MyNodeTemplate::CreateSpatialLocationCalculator => {
                build_depthai_input(
                    node_id,
                    graph,
                    "inputConfig".into(),
                    DepthaiNode::SpatialLocationCalculator,
                );
                build_depthai_input(
                    node_id,
                    graph,
                    "inputDepth".into(),
                    DepthaiNode::SpatialLocationCalculator,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "out".into(),
                    DepthaiNode::SpatialLocationCalculator,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "passthroughDepth".into(),
                    DepthaiNode::SpatialLocationCalculator,
                );
            }
            MyNodeTemplate::CreateEdgeDetector => {
                build_depthai_input(
                    node_id,
                    graph,
                    "inputConfig".into(),
                    DepthaiNode::EdgeDetector,
                );
                build_depthai_input(
                    node_id,
                    graph,
                    "inputImage".into(),
                    DepthaiNode::EdgeDetector,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "outputImage".into(),
                    DepthaiNode::EdgeDetector,
                );
            }
            MyNodeTemplate::CreateFeaureTracker => {
                build_depthai_input(
                    node_id,
                    graph,
                    "inputConfig".into(),
                    DepthaiNode::FeaureTracker,
                );
                build_depthai_input(
                    node_id,
                    graph,
                    "inputImage".into(),
                    DepthaiNode::FeaureTracker,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "outputFeatures".into(),
                    DepthaiNode::FeaureTracker,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "passthroughInputImage".into(),
                    DepthaiNode::FeaureTracker,
                );
            }
            MyNodeTemplate::CreateObjectTracker => {
                build_depthai_input(
                    node_id,
                    graph,
                    "inputDetectionFrame".into(),
                    DepthaiNode::ObjectTracker,
                );
                build_depthai_input(
                    node_id,
                    graph,
                    "inputDetections".into(),
                    DepthaiNode::ObjectTracker,
                );
                build_depthai_input(
                    node_id,
                    graph,
                    "inputTrackerFrame".into(),
                    DepthaiNode::ObjectTracker,
                );
                build_depthai_output(node_id, graph, "out".into(), DepthaiNode::ObjectTracker);
                build_depthai_output(
                    node_id,
                    graph,
                    "passthroughDetectionFrame".into(),
                    DepthaiNode::ObjectTracker,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "passthroughDetections".into(),
                    DepthaiNode::ObjectTracker,
                );
                build_depthai_output(
                    node_id,
                    graph,
                    "passthroughTrackerFrame".into(),
                    DepthaiNode::ObjectTracker,
                );
            }
            MyNodeTemplate::CreateIMU => {
                build_depthai_output(node_id, graph, "out".into(), DepthaiNode::IMU {});
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
use serde;
use serde_json;

#[derive(serde::Deserialize, serde::Serialize)]
pub enum IOKind {
    Output,
    Input,
}

#[derive(serde::Deserialize, serde::Serialize, Clone)]
pub struct IOInfo {
    blocking: bool,
    group: String,
    id: i32,
    name: String,
    #[serde(rename = "queueSize")]
    queue_size: i32,
    #[serde(rename = "type")]
    kind: i32,
    #[serde(rename = "waitForMessage")]
    wait_for_message: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Clone)]
pub struct Node {
    id: i32,
    #[serde(rename = "ioInfo")]
    io_info: Vec<((String, String), IOInfo)>,
    name: String,
    properties: Vec<i32>,

    #[serde(skip)]
    /// This is not a part of the schema dump, it's an internal field used when laying out the graph
    position: egui::Pos2,
}

impl Node {
    fn set_pos(&mut self, x: f32, y: f32) {
        self.position = egui::pos2(x, y);
    }

    fn get_pos(&self) -> egui::Pos2 {
        self.position
    }

    fn template(&self) -> MyNodeTemplate {
        match self.name.as_str() {
            "ColorCamera" => MyNodeTemplate::CreateColorCamera,
            "MonoCamera" => MyNodeTemplate::CreateMonoCamera,
            "ImageManip" => MyNodeTemplate::CreateImageManip,
            "VideoEncoder" => MyNodeTemplate::CreateVideoEncoder,

            "NeuralNetwork" => MyNodeTemplate::CreateNeuralNetwork,
            "DetectionNetwork" => MyNodeTemplate::CreateDetectionNetwork,
            "MobileNetDetectionNetwork" => MyNodeTemplate::CreateMobileNetDetectionNetwork,
            "MobileNetSpatialDetectionNetwork" => {
                MyNodeTemplate::CreateMobileNetSpatialDetectionNetwork
            }
            "YoloDetectionNetwork" => MyNodeTemplate::CreateYoloDetectionNetwork,
            "YoloSpatialDetectionNetwork" => MyNodeTemplate::CreateYoloSpatialDetectionNetwork,

            "SPIIn" => MyNodeTemplate::CreateSPIIn,
            "SPIOut" => MyNodeTemplate::CreateSPIOut,

            "XLinkIn" => MyNodeTemplate::CreateXLinkIn,
            "XLinkOut" => MyNodeTemplate::CreateXLinkOut,

            "Script" => {
                let mut input_names = Vec::new();
                let mut output_names = Vec::new();

                for ((group, _), io_info) in &self.io_info {
                    if group != "io" {
                        continue;
                    }
                    if io_info.kind == IOKind::Input as i32 || io_info.kind == 3 {
                        input_names.push(io_info.name.clone());
                    } else {
                        output_names.push(io_info.name.clone());
                    }
                }

                MyNodeTemplate::CreateScript(input_names, output_names)
            }

            "StereoDepth" => MyNodeTemplate::CreateStereoDepth,
            "SpatialLocationCalculator" => MyNodeTemplate::CreateSpatialLocationCalculator,

            "EdgeDetector" => MyNodeTemplate::CreateEdgeDetector,
            "FeaureTracker" => MyNodeTemplate::CreateFeaureTracker,
            "ObjectTracker" => MyNodeTemplate::CreateObjectTracker,
            "IMU" => MyNodeTemplate::CreateIMU,
            _ => panic!("Unknown node: {:?}", self.name),
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct Connection {
    #[serde(rename = "node1Id")]
    source_node: i32,
    #[serde(rename = "node1Output")]
    source_node_output: String,
    #[serde(rename = "node2Id")]
    dest_node: i32,
    #[serde(rename = "node2Input")]
    dest_node_input: String,
    #[serde(rename = "node2InputGroup")]
    dest_node_input_group: String,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct Schema {
    nodes: Vec<(i32, Node)>,
    connections: Vec<Connection>,
}

impl Schema {
    fn all_connections(&self, node: &Node) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|c| c.source_node == node.id || c.dest_node == node.id)
            .collect()
    }

    fn get_nodes_connected_to_outputs(&self, node: &Node) -> Vec<&Node> {
        self.connections
            .iter()
            .filter(|c| c.source_node == node.id)
            .map(|c| {
                &self
                    .nodes
                    .iter()
                    .find(|(id, _)| *id == c.dest_node)
                    .unwrap()
                    .1
            })
            .collect::<Vec<_>>()
    }

    fn update_node_rank(&self, node: i32, nodes_rank: &mut HashMap<i32, i32>) {
        println!("Updating node rank for node: {:?}", node);
        let mut connected_nodes = HashSet::new();
        let Some(node_name) = self
            .nodes
            .iter()
            .find(|(id, _)| *id == node).map(|(_, node)| node.name.clone()) else {
                return;
            };

        for ((_, _), io_info) in &self
            .nodes
            .iter()
            .find(|(id, _)| *id == node)
            .unwrap()
            .1
            .io_info
        {
            println!(
                "IO kind: {:?}, Name: {:?}, for node: {node_name:?}",
                io_info.kind, io_info.name,
            );
            if io_info.kind == IOKind::Output as i32 {
                connected_nodes.insert(io_info.id);
            }
        }
        println!(
            "Connected nodes: {:?} for node: {:?}",
            connected_nodes, node_name
        );

        let rank = nodes_rank.get(&node).unwrap_or(&0) + 1;
        for n in connected_nodes {
            if let Some(n_rank) = nodes_rank.get_mut(&n) {
                *n_rank = std::cmp::max(*n_rank, rank);
            } else {
                nodes_rank.insert(n, rank);
            }
            self.update_node_rank(n, nodes_rank);
        }
    }

    fn compute_node_rank(&self, nodes: &Vec<Node>) -> HashMap<i32, i32> {
        let mut nodes_rank = HashMap::new();
        for node in nodes {
            nodes_rank.insert(node.id, 0);
            self.update_node_rank(node.id, &mut nodes_rank);
        }
        nodes_rank
    }

    pub fn auto_layout_nodes(&mut self) {
        const START_NODES: [&str; 3] = ["ColorCamera", "XLinkIn", "MonoCamera"];
        let mut start_nodes = Vec::new();
        for (id, node) in &self.nodes {
            if START_NODES.contains(&node.name.as_str()) {
                start_nodes.push(node.clone());
            }
        }

        let node_rank = self.compute_node_rank(&start_nodes);
        println!("Node rank: {:?}", node_rank);
        let mut rank_map = HashMap::new();
        for (id, rank) in node_rank.iter() {
            let Some(node_id) = self.nodes
            .iter()
            .find(|(node_id, _)| node_id == id).map(|(node_id, _)| *node_id) else {
                continue;
            };
            rank_map.entry(rank).or_insert(Vec::new()).push(node_id);
        }

        let mut current_x = 0.0;
        let node_height = 120.0;
        println!("Rank map: {:?}", rank_map);
        for rank in 0..rank_map.len() {
            let ranked_nodes = rank_map.get(&(rank as i32)).unwrap();
            let max_width = 150.0;
            current_x += max_width;
            let mut current_y = 0.0;
            for (idx, node) in ranked_nodes.iter().enumerate() {
                let node = &mut self.nodes.iter_mut().find(|(id, _)| id == node).unwrap().1;
                let dy = node.get_pos().y.max(node_height);
                current_y += if idx == 0 { 0.0 } else { dy };
                node.set_pos(current_x, current_y);
                current_y += dy * 0.5 + 10.0;
            }
        }
    }
}

impl NodeGraphExample {
    /// Loads json schema dump into the graph
    fn create_nodes_from_dump(&mut self) {
        const DUMP_FILE: &str = "schema.json";
        let dump = std::fs::read_to_string(DUMP_FILE).unwrap();
        let mut schema: Schema = serde_json::from_str(dump.as_str()).unwrap();
        schema.auto_layout_nodes();
        let graph = &mut self.state.graph;

        let mut node_id_to_graph_node_id = HashMap::new();
        for (id, node) in schema.nodes {
            let template = node.template();
            let graph_node = graph.add_node(
                node.name.clone(),
                MyNodeData {
                    template: template.clone(),
                },
                |g, node_id| template.build_node(g, &mut self.user_state, node_id),
            );
            node_id_to_graph_node_id.insert(id, graph_node.clone());
            println!(
                "Node: {:?} : Outputs: {:?}",
                node.name,
                graph
                    .nodes
                    .get(graph_node)
                    .unwrap()
                    .outputs
                    .iter()
                    .map(|(name, _)| name.clone())
            );
            self.state.node_positions.insert(graph_node, node.get_pos());
            self.state.node_order.push(graph_node);
        }
        // Now create connections
        for connection in schema.connections {
            println!("Creating connection: {connection:?}");

            let source_node = node_id_to_graph_node_id
                .get(&connection.source_node)
                .unwrap()
                .clone();
            let dest_node = node_id_to_graph_node_id
                .get(&connection.dest_node)
                .unwrap()
                .clone();

            let source_node_output = graph
                .nodes
                .get(source_node)
                .unwrap()
                .outputs
                .iter()
                .find(|(name, conn_id)| {
                    println!(
                        "Checking src {name:?} == {:?}",
                        connection.source_node_output
                    );
                    name == &connection.source_node_output
                })
                .unwrap()
                .1;
            let dest_node_input = graph
                .nodes
                .get(dest_node)
                .unwrap()
                .inputs
                .iter()
                .find(|(name, conn_id)| {
                    println!("Checking dst {name:?} == {:?}", connection.dest_node_input);
                    name == &connection.dest_node_input
                })
                .unwrap()
                .1;

            graph.add_connection(source_node_output, dest_node_input);
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
            self.create_nodes_from_dump();
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
            let input = evaluator.input_queue("in")?;
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
