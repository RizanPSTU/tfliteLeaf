#import tensorflow as tf
#
#graph_def_file = "./main/f.pb"
#input_arrays = ["ImageTensor"]
#output_arrays = ["SemanticPredictions"]
#
#converter = tf.lite.TFLiteConverter.from_frozen_graph(
#  graph_def_file, input_arrays, output_arrays , )
#tflite_model = converter.convert()
#open("converted_model.tflite", "wb").write(tflite_model)




import tensorflow as tf

def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
        
    with open("name.txt", "w") as a:
       for op in graph.get_operations():
         print(op.values())
         a.write(str(op.name)+'          value:  '+str(op.values())+'\n') 

printTensors("./main/f.pb")