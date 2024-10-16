#inference_session_from_url <- function(url, tokenizer_path, num_threads = 1) {
#    ORTSession$new_from_url(num_threads,url, tokenizer_path)
#}

#' Initialize a new Inference Session from a saved model
#'
#' @param model_path the path to where the onnx model is stored
#'
#' @param tokenizer_path the path to the tokenizer is stored
#'
#' @param num_thread number of threads for inference to run on (default 1)
#'
#' @export
new_inference_session <- function(directory, model_name = "model.onnx", tokenizer_name = "tokenizer.json") {
    model_path <- file.path(directory, model_name)
    tokenizer_path <- file.path(directory, tokenizer_name)
    ORTSession$new_from_path(model_path, tokenizer_path)
}


#' Run a Model on Given Inputs
#'
#' @param session an Inference Session object
#'
#' @param inputs a vector of string inputs you want to calculate the embeddings for
#'
#' @export
generate_embeddings <- function(session, inputs, output_index = 0, mean_pooling_needed = FALSE) {
    session$run_model(inputs, output_index, mean_pooling_needed)
}

