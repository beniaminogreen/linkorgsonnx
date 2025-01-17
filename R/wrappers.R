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
    ort_session_pointer <- ORTSession$new_from_path(model_path, tokenizer_path)

    output <- list(
                   ort_session_pointer = ort_session_pointer,
                   model_path = model_path,
                   tokenizer_path = tokenizer_path
                   )
    class(output) <- "ort_session"

    return(output)
}


#' Generate Embeddings
#'
#' @param session an Inference Session object
#'
#' @param inputs a vector of string inputs you want to calculate the embeddings for
#'
#' @export
generate_embeddings <- function(session, inputs, output_index = 0, mean_pooling_needed = FALSE) {
    session$ort_session_pointer$run_model(inputs, output_index, mean_pooling_needed)
}


#' @export
print.ort_session <- function(x, ...) {
    cat("An ORT inference session\n")
    cat("Using the model found at:\n\t", x$model_path, "\n")
    cat("And the tokenizer found at:\n\t", x$tokenizer_path, "\n")
}
