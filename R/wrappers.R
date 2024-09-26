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
inference_session_from_file <- function(model_path,tokenizer_path,  num_threads = 1) {
    ORTSession$new_from_path(num_threads,model_path, tokenizer_path)
}


#' Run a Model on Given Inputs
#'
#' @param session an Inference Session object
#'
#' @param inputs a vector of string inputs you want to calculate the embeddings for
#'
#' @export
run_model <- function(session, inputs) {
    session$run_model(inputs)
}

