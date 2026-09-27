#include <stdio.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// O ficheiro gerado pelo comando xxd
#include "modelo_medidor.h"

// Memória de trabalho (Tensor Arena)
constexpr int kTensorArenaSize = 5000 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int main(int argc, char* argv[]) {
    MicroPrintf("-> Iniciando Teste do Modelo no WSL (Linux Nativo)\n");

    const tflite::Model* model = tflite::GetModel(modelo_medidor_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("ERRO: Versao do schema %d nao suportada. Esperada: %d",
            model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    // O novo padrão: declaramos apenas as operações matemáticas que a sua CNN usa
    tflite::MicroMutableOpResolver<15> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu();
    resolver.AddAdd();
    resolver.AddConcatenation();
    resolver.AddLogistic();
    resolver.AddResizeBilinear();
    resolver.AddResizeNearestNeighbor();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddPad();
    resolver.AddTransposeConv();

    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        MicroPrintf("ERRO FATAL: Falha ao alocar memoria (Arena pequena demais).");
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    MicroPrintf("-> Modelo pronto!");
    MicroPrintf("   - Entrada esperada: %d bytes (Tipo: %d)", input->bytes, input->type);
    MicroPrintf("   - Saida esperada: %d bytes (Tipo: %d)\n", output->bytes, output->type);

    MicroPrintf("-> Executando inferencia com matriz de teste...");
    for (int i = 0; i < input->bytes; i++) {
        input->data.int8[i] = 0;
    }

    if (interpreter.Invoke() != kTfLiteOk) {
        MicroPrintf("ERRO: Falha ao invocar o modelo.");
        return 1;
    }

    MicroPrintf("-> Inferencia concluida com sucesso!");
    MicroPrintf("-> Amostra Saida (Pixel 0): %d", output->data.int8[0]);

    return 0;
}