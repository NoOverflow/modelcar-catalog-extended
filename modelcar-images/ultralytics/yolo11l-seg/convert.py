from ultralytics import YOLO

model = YOLO("/tmp/yolo.pt")

# Retrieve metadata during export. Metadata needs to be added to config.pbtxt. See next section.
metadata = []


def export_cb(exporter):
    metadata.append(exporter.metadata)


model.add_callback("on_export_end", export_cb)

# Export the model
onnx_file = model.export(format="onnx", dynamic=True)

data = """
parameters {
  key: "metadata"
  value {
    string_value: "%s"
  }
}
""" % metadata[0]

with open("config.pbtxt", "w") as f:
    f.write(data)