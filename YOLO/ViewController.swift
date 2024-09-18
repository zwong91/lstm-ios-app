//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Main View Controller for Ultralytics YOLO App
//  This file is part of the Ultralytics YOLO app, enabling real-time object detection using YOLOv8 models on iOS devices.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This ViewController manages the app's main screen, handling video capture, model selection, detection visualization,
//  and user interactions. It sets up and controls the video preview layer, handles model switching via a segmented control,
//  manages UI elements like sliders for confidence and IoU thresholds, and displays detection results on the video feed.
//  It leverages CoreML, Vision, and AVFoundation frameworks to perform real-time object detection and to interface with
//  the device's camera.

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision
import Foundation
import Accelerate

var mlModel = try! lstm_coord_0914_int8(configuration: .init()).model

class ViewController: UIViewController {
  @IBOutlet var videoPreview: UIView!
  @IBOutlet var View0: UIView!
  @IBOutlet var segmentedControl: UISegmentedControl!
  @IBOutlet var playButtonOutlet: UIBarButtonItem!
  @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
  @IBOutlet var slider: UISlider!
  @IBOutlet var sliderConf: UISlider!
  @IBOutlet weak var sliderConfLandScape: UISlider!
  @IBOutlet var sliderIoU: UISlider!
  @IBOutlet weak var sliderIoULandScape: UISlider!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelZoom: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var labelSlider: UILabel!
  @IBOutlet weak var labelSliderConf: UILabel!
  @IBOutlet weak var labelSliderConfLandScape: UILabel!
  @IBOutlet weak var labelSliderIoU: UILabel!
  @IBOutlet weak var labelSliderIoULandScape: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var forcus: UIImageView!
  @IBOutlet weak var toolBar: UIToolbar!

  let selection = UISelectionFeedbackGenerator()
  var session: AVCaptureSession!
  var videoCapture: VideoCapture!
  var currentBuffer: CVPixelBuffer?
  var framesDone = 0
  var t0 = 0.0  // inference start
  var t1 = 0.0  // inference dt
  var t2 = 0.0  // inference dt smoothed
  var t3 = CACurrentMediaTime()  // FPS start
  var t4 = 0.0  // FPS dt smoothed
  // var cameraOutput: AVCapturePhotoOutput!

  // Developer mode
  let developerMode = UserDefaults.standard.bool(forKey: "developer_mode")  // developer mode selected in settings
  let save_detections = false  // write every detection to detections.txt
  let save_frames = false  // write every frame to frames.txt


  override func viewDidLoad() {
    super.viewDidLoad()
    slider.value = 30
    setLabels()
    setUpBoundingBoxViews()
    setUpOrientationChangeNotification()
    // startVideo()
    // setModel()
  }

  override func viewWillTransition(
    to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator
  ) {
    super.viewWillTransition(to: size, with: coordinator)

    if size.width > size.height {
      labelSliderConf.isHidden = true
      sliderConf.isHidden = true
      labelSliderIoU.isHidden = true
      sliderIoU.isHidden = true
      toolBar.setBackgroundImage(UIImage(), forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(UIImage(), forToolbarPosition: .any)

      labelSliderConfLandScape.isHidden = false
      sliderConfLandScape.isHidden = false
      labelSliderIoULandScape.isHidden = false
      sliderIoULandScape.isHidden = false

    } else {
      labelSliderConf.isHidden = false
      sliderConf.isHidden = false
      labelSliderIoU.isHidden = false
      sliderIoU.isHidden = false
      toolBar.setBackgroundImage(nil, forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(nil, forToolbarPosition: .any)

      labelSliderConfLandScape.isHidden = true
      sliderConfLandScape.isHidden = true
      labelSliderIoULandScape.isHidden = true
      sliderIoULandScape.isHidden = true
    }
    self.videoCapture.previewLayer?.frame = CGRect(
      x: 0, y: 0, width: size.width, height: size.height)

  }

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    //videoCapture.updateVideoOrientation()
  }

  @IBAction func vibrate(_ sender: Any) {
    selection.selectionChanged()
  }

    

    
  @IBAction func indexChanged(_ sender: Any) {
    selection.selectionChanged()
    activityIndicator.startAnimating()

    /// Switch model
    switch segmentedControl.selectedSegmentIndex {
    case 0:
      self.labelName.text = "lstm-test"
      mlModel = try! lstm_coord_0914_int8(configuration: .init()).model
      print("Setting labelName to 'lstm-test' and initializing mlModel with lstm_coord_0914_int8")
    default:
      break
    }
      
    /// 处理所有的逻辑

      
      // 定义类型别名
      typealias Detection = [[Double]]

      // 转换数组中的 Double 到 Int
      func convertToIntArray(from doubleArray: [Double]) -> [Int] {
          return doubleArray.map { Int($0) }
      }

      // 检测数据示例
      let cardDetections: [String: [[Double]]] = [
          "Tc": [
              [552, 1130, 601, 1237, 0.6396484475],
              [554, 1131, 602, 1240, 0.6396484575],
              [555, 1132, 603, 1241, 0.6396484675],
              [556, 1133, 604, 1242, 0.6396484175],
          ],
          "Td": [
              [552, 1130, 651, 1257, 0.6396484475],
              [554, 1131, 662, 1260, 0.6396484575],
              [555, 1132, 663, 1271, 0.6396484675],
              [556, 1133, 674, 1282, 0.6396484175],
          ]
      ]

      // 打印所有检测数据
      func printDetections(_ cardDetections: [String: [[Double]]]) {
          for (key, value) in cardDetections {
              print("\(key):")
              for detection in value {
                  print(detection)
              }
              print("\n")
          }
      }

      printDetections(cardDetections)

      // 创建 LSTM 的输入数据
      func preparePredictSequenceData(cardDetections: [[Double]], sequenceLength: Int) -> [[[Double]]] {
          var X: [[[Double]]] = []
          for i in 0...(cardDetections.count - sequenceLength) {
              let inputSeq = (0..<sequenceLength).map { j -> [Double] in
                  let cardBoxes = cardDetections[i + j]
                  let cx = (cardBoxes[0] + cardBoxes[2]) / 2
                  let cy = (cardBoxes[1] + cardBoxes[3]) / 2
                  return [cx, cy]
              }
              X.append(inputSeq)
          }
          return X
      }

      // 加载 CoreML 模型和 scaler
      func loadModel(from fileName: String, extension: String) -> MLModel {
          guard let url = Bundle.main.url(forResource: fileName, withExtension: `extension`),
                let model = try? MLModel(contentsOf: url) else {
              fatalError("Model \(fileName) not found in bundle")
          }
          return model
      }

      func loadScaler(from file: String) -> (min: [Double], max: [Double]) {
          guard let url = Bundle.main.url(forResource: file, withExtension: "json"),
                let data = try? Data(contentsOf: url),
                let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: [Double]] else {
              fatalError("Failed to load scaler file \(file)")
          }
          let min = json["min"] ?? []
          let max = json["max"] ?? []
          return (min, max)
      }

      // 归一化函数
      func normalizeData(_ data: [[[Double]]], min: [Double], max: [Double]) -> [[[Double]]] {
          return data.map { matrix in
              matrix.map { point in
                  zip(point, zip(min, max)).map { (value, range) in
                      (value - range.0) / (range.1 - range.0)
                  }
              }
          }
      }

      func denormalizeData(_ data: [Double], min: [Double], max: [Double]) -> [Double] {
          return zip(data, zip(min, max)).map { (value, range) in
              value * (range.1 - range.0) + range.0
          }
      }

      // 预测函数
      func predict(for cardType: String, model: MLModel, cardDetections: [String: [[Double]]], timeSteps: Int, scalerX: (min: [Double], max: [Double]), scalerY: (min: [Double], max: [Double])) -> [[[Double]]]? {
          guard var selectedObjects = cardDetections[cardType] else {
              print("No detections for card type \(cardType)")
              return nil
          }
          
          // 对数据进行填充或截取
          if selectedObjects.count < timeSteps {
              let firstElement = selectedObjects.first ?? [Double](repeating: 0.0, count: 0)
              let padding = Array(repeating: firstElement, count: timeSteps - selectedObjects.count)
              selectedObjects = padding + selectedObjects
          } else {
              selectedObjects = Array(selectedObjects.suffix(timeSteps))
          }
          
          var X = preparePredictSequenceData(cardDetections: selectedObjects, sequenceLength: timeSteps)
          print("Initial predict sequenceData: \(X)")
          var predictionsList: [[[Double]]] = []
          
          for _ in 0..<10 {
              let X_scaled = normalizeData(X, min: scalerX.min, max: scalerX.max)
              let inputArray = try? MLMultiArray(shape: [NSNumber(value: X.count), NSNumber(value: timeSteps), NSNumber(value: X.first?.first?.count ?? 0)], dataType: .double)
              
              // 将 X_scaled 填充到 MLMultiArray 中
              for (index, value) in X_scaled.flatMap({ $0.flatMap { $0 } }).enumerated() {
                  inputArray?[index] = NSNumber(value: value)
              }
              
              guard let inputData = inputArray,
                    let predictionInput = try? MLDictionaryFeatureProvider(dictionary: ["input_2": inputData]),
                    let output = try? model.prediction(from: predictionInput),
                    let yPredScaled = output.featureValue(for: "Identity")?.multiArrayValue else {
                  print("Prediction failed for card type \(cardType)")
                  continue
              }
              
              // 反标准化预测结果
              let yPred = (0..<yPredScaled.count).map { yPredScaled[$0].doubleValue }
              let yPredDenormalized = denormalizeData(yPred, min: scalerY.min, max: scalerY.max)
              
              predictionsList.append(yPredDenormalized.map { [$0] })
              // 确保 yPredDenormalized 至少有两个值 [cx, cy]
              if yPredDenormalized.count >= 2 {
                  let newPrediction = [yPredDenormalized[0], yPredDenormalized[1]]  // 创建新的时间步
                  
                  // 删除 X[0] 中的第一个 [cx, cy]
                  if !X.isEmpty && !X[0].isEmpty {
                      X[0].removeFirst()
                  }
                  
                  // 添加新的预测结果到 X 的末尾
                  if X.count > 0 {
                      X[0].append(newPrediction)  // 追加新的预测到现有时间步中
                  }
                  
                  // 确保 X 的长度保持为 timeSteps
                  if X.count > timeSteps {
                      X.removeFirst()
                  }
                  
                  print("Updated predict sequenceData: \(X)")
              }
          }
          
          return predictionsList
      }

      // 主执行逻辑
      let coremlModel = loadModel(from: "lstm_coord_0914_int8", extension: "mlmodelc")
      let scalerX = loadScaler(from: "scaler_X")
      let scalerY = loadScaler(from: "scaler_y")

      let groupTypes = [
          ["Tc", "Td"],
          ["9c", "2c"],
      ]

      for (idx, types) in groupTypes.enumerated() {
          var predictionsList: [[[Double]]] = []
          
          for cardType in types {
              if let predictions = predict(for: cardType, model: coremlModel, cardDetections: cardDetections, timeSteps: 5, scalerX: scalerX, scalerY: scalerY) {
                  predictionsList.append(contentsOf: predictions)
              }
          }
          
          print("第 \(idx + 1) 组预测结果:\n")
          if predictionsList.count != 20 {
              print("预测总数:\(predictionsList.count) 牌类型 \(types[0]) 和 \(types[1]) 无预测结果")
              continue
          }
          
          var type0 = 0
          var type1 = 0
          for iteration in 5..<10 {
              let prediction1 = predictionsList[iteration]
              let prediction2 = predictionsList[iteration + 10]
              
              let flatPrediction1 = prediction1.flatMap { $0 }
              let flatPrediction2 = prediction2.flatMap { $0 }
              
              let futureCy1 = flatPrediction1.count >= 2 ? flatPrediction1[1] : 0
              let futureCy2 = flatPrediction2.count >= 2 ? flatPrediction2[1] : 0
              print("牌类型 \(types[0]) 和 \(types[1]) 的预测结果: \(futureCy1), vs \(futureCy2)")
              if futureCy1 > futureCy2 {
                  type0 += 1
              } else {
                  type1 += 1
              }
          }
          
          let sortedTypes = [(types[0], type0), (types[1], type1)].sorted { $0.1 > $1.1 }
          print("组 \(idx + 1) 的排序为 \(sortedTypes)")
      }


      
    setUpBoundingBoxViews()
    activityIndicator.stopAnimating()
  }
  /// Update thresholds from slider values
  @IBAction func sliderChanged(_ sender: Any) {
    let conf = Double(round(100 * sliderConf.value)) / 100
    let iou = Double(round(100 * sliderIoU.value)) / 100
    self.labelSliderConf.text = String(conf) + " Confidence Threshold"
    self.labelSliderIoU.text = String(iou) + " IoU Threshold"
  }

  @IBAction func takePhoto(_ sender: Any?) {
    let t0 = DispatchTime.now().uptimeNanoseconds

    // 1. captureSession and cameraOutput
    // session = videoCapture.captureSession  // session = AVCaptureSession()
    // session.sessionPreset = AVCaptureSession.Preset.photo
    // cameraOutput = AVCapturePhotoOutput()
    // cameraOutput.isHighResolutionCaptureEnabled = true
    // cameraOutput.isDualCameraDualPhotoDeliveryEnabled = true
    // print("1 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)

    // 2. Settings
    let settings = AVCapturePhotoSettings()
    // settings.flashMode = .off
    // settings.isHighResolutionPhotoEnabled = cameraOutput.isHighResolutionCaptureEnabled
    // settings.isDualCameraDualPhotoDeliveryEnabled = self.videoCapture.cameraOutput.isDualCameraDualPhotoDeliveryEnabled

    // 3. Capture Photo
    usleep(20_000)  // short 10 ms delay to allow camera to focus
    self.videoCapture.cameraOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    print("3 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)
  }

  @IBAction func logoButton(_ sender: Any) {
    selection.selectionChanged()
    if let link = URL(string: "https://www.ultralytics.com") {
      UIApplication.shared.open(link)
    }
  }

  func setLabels() {
    self.labelName.text = "xxxxxx"
    self.labelVersion.text = "Version " + UserDefaults.standard.string(forKey: "app_version")!
  }

  @IBAction func playButton(_ sender: Any) {
    selection.selectionChanged()
    self.videoCapture.start()
    playButtonOutlet.isEnabled = false
    pauseButtonOutlet.isEnabled = true
  }

  @IBAction func pauseButton(_ sender: Any?) {
    selection.selectionChanged()
    self.videoCapture.stop()
    playButtonOutlet.isEnabled = true
    pauseButtonOutlet.isEnabled = false
  }

  @IBAction func switchCameraTapped(_ sender: Any) {
    self.videoCapture.captureSession.beginConfiguration()
    let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput
    self.videoCapture.captureSession.removeInput(currentInput!)
    // let newCameraDevice = currentInput?.device == .builtInWideAngleCamera ? getCamera(with: .front) : getCamera(with: .back)

    let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)!
    guard let videoInput1 = try? AVCaptureDeviceInput(device: device) else {
      return
    }

    self.videoCapture.captureSession.addInput(videoInput1)
    self.videoCapture.captureSession.commitConfiguration()
  }

  // share image
  @IBAction func shareButton(_ sender: Any) {
    selection.selectionChanged()
    let settings = AVCapturePhotoSettings()
    self.videoCapture.cameraOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
  }

  // share screenshot
  @IBAction func saveScreenshotButton(_ shouldSave: Bool = true) {
    // let layer = UIApplication.shared.keyWindow!.layer
    // let scale = UIScreen.main.scale
    // UIGraphicsBeginImageContextWithOptions(layer.frame.size, false, scale);
    // layer.render(in: UIGraphicsGetCurrentContext()!)
    // let screenshot = UIGraphicsGetImageFromCurrentImageContext()
    // UIGraphicsEndImageContext()

    // let screenshot = UIApplication.shared.screenShot
    // UIImageWriteToSavedPhotosAlbum(screenshot!, nil, nil, nil)
  }

  let maxBoundingBoxViews = 100
  var boundingBoxViews = [BoundingBoxView]()
  var colors: [String: UIColor] = [:]

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }

  }

  func startVideo() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self

    videoCapture.setUp(sessionPreset: .photo) { success in
      // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.videoCapture.previewLayer?.frame = self.videoPreview.bounds  // resize preview layer
        }

        // Add the bounding box layers to the UI, on top of the video preview.
        for box in self.boundingBoxViews {
          box.addToLayer(self.videoPreview.layer)
        }

        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  func predict(sampleBuffer: CMSampleBuffer) {
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer

      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor,
      // so in most cases Vision needs to rotate it for the model to work as expected.
      let imageOrientation: CGImagePropertyOrientation
      switch UIDevice.current.orientation {
      case .portrait:
        imageOrientation = .up
      case .portraitUpsideDown:
        imageOrientation = .down
      case .landscapeLeft:
        imageOrientation = .up
      case .landscapeRight:
        imageOrientation = .up
      case .unknown:
        imageOrientation = .up
      default:
        imageOrientation = .up
      }

      // Invoke a VNRequestHandler with that image
      let handler = VNImageRequestHandler(
        cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
      if UIDevice.current.orientation != .faceUp {  // stop if placed down on a table
        t0 = CACurrentMediaTime()  // inference start
        do {
          //try handler.perform([visionRequest])
        } catch {
          print(error)
        }
        t1 = CACurrentMediaTime() - t0  // inference dt
      }

      currentBuffer = nil
    }
  }

  func processObservations(for request: VNRequest, error: Error?) {
    DispatchQueue.main.async {
      if let results = request.results as? [VNRecognizedObjectObservation] {
        self.show(predictions: results)
      } else {
        self.show(predictions: [])
      }

      // Measure FPS
      if self.t1 < 10.0 {  // valid dt
        self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
      }
      self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
      self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
      self.t3 = CACurrentMediaTime()
    }
  }

  // Save text file
  func saveText(text: String, file: String = "saved.txt") {
    if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
      let fileURL = dir.appendingPathComponent(file)

      // Writing
      do {  // Append to file if it exists
        let fileHandle = try FileHandle(forWritingTo: fileURL)
        fileHandle.seekToEndOfFile()
        fileHandle.write(text.data(using: .utf8)!)
        fileHandle.closeFile()
      } catch {  // Create new file and write
        do {
          try text.write(to: fileURL, atomically: false, encoding: .utf8)
        } catch {
          print("no file written")
        }
      }

      // Reading
      // do {let text2 = try String(contentsOf: fileURL, encoding: .utf8)} catch {/* error handling here */}
    }
  }

  // Save image file
  func saveImage() {
    let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    let fileURL = dir!.appendingPathComponent("saved.jpg")
    let image = UIImage(named: "ultralytics_yolo_logotype.png")
    FileManager.default.createFile(
      atPath: fileURL.path, contents: image!.jpegData(compressionQuality: 0.5), attributes: nil)
  }

  // Return hard drive space (GB)
  func freeSpace() -> Double {
    let fileURL = URL(fileURLWithPath: NSHomeDirectory() as String)
    do {
      let values = try fileURL.resourceValues(forKeys: [
        .volumeAvailableCapacityForImportantUsageKey
      ])
      return Double(values.volumeAvailableCapacityForImportantUsage!) / 1E9  // Bytes to GB
    } catch {
      print("Error retrieving storage capacity: \(error.localizedDescription)")
    }
    return 0
  }

  // Return RAM usage (GB)
  func memoryUsage() -> Double {
    var taskInfo = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
      $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }
    if kerr == KERN_SUCCESS {
      return Double(taskInfo.resident_size) / 1E9  // Bytes to GB
    } else {
      return 0
    }
  }

  func show(predictions: [VNRecognizedObjectObservation]) {
    let width = videoPreview.bounds.width  // 375 pix
    let height = videoPreview.bounds.height  // 812 pix
    var str = ""

    // ratio = videoPreview AR divided by sessionPreset AR
    var ratio: CGFloat = 1.0
    if videoCapture.captureSession.sessionPreset == .photo {
      ratio = (height / width) / (4.0 / 3.0)  // .photo
    } else {
      ratio = (height / width) / (16.0 / 9.0)  // .hd4K3840x2160, .hd1920x1080, .hd1280x720 etc.
    }

    // date
    let date = Date()
    let calendar = Calendar.current
    let hour = calendar.component(.hour, from: date)
    let minutes = calendar.component(.minute, from: date)
    let seconds = calendar.component(.second, from: date)
    let nanoseconds = calendar.component(.nanosecond, from: date)
    let sec_day =
      Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9  // seconds in the day

    self.labelSlider.text =
      String(predictions.count) + " items (max " + String(Int(slider.value)) + ")"
    for i in 0..<boundingBoxViews.count {
      if i < predictions.count && i < Int(slider.value) {
        let prediction = predictions[i]

        var rect = prediction.boundingBox  // normalized xywh, origin lower left
        switch UIDevice.current.orientation {
        case .portraitUpsideDown:
          rect = CGRect(
            x: 1.0 - rect.origin.x - rect.width,
            y: 1.0 - rect.origin.y - rect.height,
            width: rect.width,
            height: rect.height)
        case .landscapeLeft:
          rect = CGRect(
            x: rect.origin.x,
            y: rect.origin.y,
            width: rect.width,
            height: rect.height)
        case .landscapeRight:
          rect = CGRect(
            x: rect.origin.x,
            y: rect.origin.y,
            width: rect.width,
            height: rect.height)
        case .unknown:
          print("The device orientation is unknown, the predictions may be affected")
          fallthrough
        default: break
        }

        if ratio >= 1 {  // iPhone ratio = 1.218
          let offset = (1 - ratio) * (0.5 - rect.minX)
          let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
          rect = rect.applying(transform)
          rect.size.width *= ratio
        } else {  // iPad ratio = 0.75
          let offset = (ratio - 1) * (0.5 - rect.maxY)
          let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
          rect = rect.applying(transform)
          ratio = (height / width) / (3.0 / 4.0)
          rect.size.height /= ratio
        }

        // Scale normalized to pixels [375, 812] [width, height]
        rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))

        // The labels array is a list of VNClassificationObservation objects,
        // with the highest scoring class first in the list.
        let bestClass = prediction.labels[0].identifier
        let confidence = prediction.labels[0].confidence
        // print(confidence, rect)  // debug (confidence, xywh) with xywh origin top left (pixels)
        let label = String(format: "%@ %.1f", bestClass, confidence * 100)
        let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
        // Show the bounding box.
        boundingBoxViews[i].show(
          frame: rect,
          label: label,
          color: colors[bestClass] ?? UIColor.white,
          alpha: alpha)  // alpha 0 (transparent) to 1 (opaque) for conf threshold 0.2 to 1.0)

        if developerMode {
          // Write
          if save_detections {
            str += String(
              format: "%.3f %.3f %.3f %@ %.2f %.1f %.1f %.1f %.1f\n",
              sec_day, freeSpace(), UIDevice.current.batteryLevel, bestClass, confidence,
              rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
          }

          // Action trigger upon detection
          // if false {
          //     if (bestClass == "car") {  // "cell phone", "car", "person"
          //         self.takePhoto(nil)
          //         // self.pauseButton(nil)
          //         sleep(2)
          //     }
          // }
        }
      } else {
        boundingBoxViews[i].hide()
      }
    }

    // Write
    if developerMode {
      if save_detections {
        saveText(text: str, file: "detections.txt")  // Write stats for each detection
      }
      if save_frames {
        str = String(
          format: "%.3f %.3f %.3f %.3f %.1f %.1f %.1f\n",
          sec_day, freeSpace(), memoryUsage(), UIDevice.current.batteryLevel,
          self.t1 * 1000, self.t2 * 1000, 1 / self.t4)
        saveText(text: str, file: "frames.txt")  // Write stats for each image
      }
    }

    // Debug
    // print(str)
    // print(UIDevice.current.identifierForVendor!)
    // saveImage()
  }

  // Pinch to Zoom Start ---------------------------------------------------------------------------------------------
  let minimumZoom: CGFloat = 1.0
  let maximumZoom: CGFloat = 10.0
  var lastZoomFactor: CGFloat = 1.0

  @IBAction func pinch(_ pinch: UIPinchGestureRecognizer) {
    let device = videoCapture.captureDevice

    // Return zoom value between the minimum and maximum zoom values
    func minMaxZoom(_ factor: CGFloat) -> CGFloat {
      return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
    }

    func update(scale factor: CGFloat) {
      do {
        try device.lockForConfiguration()
        defer {
          device.unlockForConfiguration()
        }
        device.videoZoomFactor = factor
      } catch {
        print("\(error.localizedDescription)")
      }
    }

    let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
    switch pinch.state {
    case .began, .changed:
      update(scale: newScaleFactor)
      self.labelZoom.text = String(format: "%.2fx", newScaleFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .title2)
    case .ended:
      lastZoomFactor = minMaxZoom(newScaleFactor)
      update(scale: lastZoomFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
    default: break
    }
  }  // Pinch to Zoom End --------------------------------------------------------------------------------------------
}  // ViewController class End

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
    predict(sampleBuffer: sampleBuffer)
  }
}

// Programmatically save image
extension ViewController: AVCapturePhotoCaptureDelegate {
  func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    if let error = error {
      print("error occurred : \(error.localizedDescription)")
    }
    if let dataImage = photo.fileDataRepresentation() {
      let dataProvider = CGDataProvider(data: dataImage as CFData)
      let cgImageRef: CGImage! = CGImage(
        jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true,
        intent: .defaultIntent)
      var orientation = CGImagePropertyOrientation.right
      switch UIDevice.current.orientation {
      case .landscapeLeft:
        orientation = .up
      case .landscapeRight:
        orientation = .down
      default:
        break
      }
      var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
      if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
        let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent)
      {
        image = UIImage(cgImage: cgImage)
      }
      let imageView = UIImageView(image: image)
      imageView.contentMode = .scaleAspectFill
      imageView.frame = videoPreview.frame
      let imageLayer = imageView.layer
      videoPreview.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

      let bounds = UIScreen.main.bounds
      UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
      self.View0.drawHierarchy(in: bounds, afterScreenUpdates: true)
      let img = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext()
      imageLayer.removeFromSuperlayer()
      let activityViewController = UIActivityViewController(
        activityItems: [img!], applicationActivities: nil)
      activityViewController.popoverPresentationController?.sourceView = self.View0
      self.present(activityViewController, animated: true, completion: nil)
      //
      //            // Save to camera roll
      //            UIImageWriteToSavedPhotosAlbum(img!, nil, nil, nil);
    } else {
      print("AVCapturePhotoCaptureDelegate Error")
    }
  }
}
