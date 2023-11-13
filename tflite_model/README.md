

TensorFlow 2.x 是一个与 TensorFlow 1.x 使用体验完全不同的框架，同时在编程风格、函数接口设计等上也大相径庭， 为确保高版本的 TensorFlow 支持低版本的代码，升级脚本加入了 compat.v1 模块，可以支持TensorFlow 1.x 部分功能的使用



f.keras 将作为 TensorFlow 2.x 版本的唯一高层接口



TF2.x的接口有好多算子不知道怎么构造，待解决



可以先通过TF1.x 去构造单算子









构造的单算子有一些量化参数可能不满足要求，可以把构建的量化模型转成 json，修改参数，然后再转回模型



# flatc使用方法

#### tflite to JSON

```bash
./flatc -t --strict-json --defaults-json -o model_dir ./schema.fbs -- model_dir/model_name.tflite
```

#### JSON to tflite

```bash
./flatc -o workdir -b ./schema.fbs model_dir/model_fname.json
```

