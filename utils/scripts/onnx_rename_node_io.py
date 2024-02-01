import onnx
import argparse


def main(args):
    onnx_model = onnx.load(args.model_path)

    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].input)):
            if onnx_model.graph.node[i].input[j] in [args.old_name]:
                print('-' * 60)
                print(onnx_model.graph.node[i].name)
                print(onnx_model.graph.node[i].input)
                print(onnx_model.graph.node[i].output)

                onnx_model.graph.node[i].input[j] = str(args.new_name)

        for j in range(len(onnx_model.graph.node[i].output)):
            if onnx_model.graph.node[i].output[j] in [args.old_name]:
                print('-' * 60)
                print(onnx_model.graph.node[i].name)
                print(onnx_model.graph.node[i].input)
                print(onnx_model.graph.node[i].output)

                onnx_model.graph.node[i].output[j] = str(args.new_name)

    for i in range(len(onnx_model.graph.input)):
        if onnx_model.graph.input[i].name in [args.old_name]:
            print('-' * 60)
            print(f"old: {onnx_model.graph.input[i]}")
            onnx_model.graph.input[i].name = str(args.new_name)
            print(f"new: {onnx_model.graph.input[i]}")

    for i in range(len(onnx_model.graph.output)):
        if onnx_model.graph.output[i].name in [args.old_name]:
            print('-' * 60)
            print(f"old: {onnx_model.graph.input[i]}")
            onnx_model.graph.input[i].name = str(args.new_name)
            print(f"new: {onnx_model.graph.input[i]}")
    onnx.checker.check_model(onnx_model)
    print(f'Model with new input name {args.new_name} has been checked')
    onnx.save(onnx_model, args.model_path)
    print(f'Model with new input name {args.new_name} has been saved to {args.model_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--old_name", default='', help="Node old name")
    parser.add_argument("--new_name", default='', help="Node new name")
    argz = parser.parse_args()
    main(argz)
