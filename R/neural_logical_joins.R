#' Fuzzy Neural joins
#'
#' @param a,b The two dataframes to join.
#'
#' @param model a pointer to the ort inference session running the model used
#' to generate embeddings
#'
#' @param by A named vector indicating which columns to join on. Format should
#'   be the same as dplyr: `by = c("column_name_in_df_a" =
#'   "column_name_in_df_b")`, but two columns must be specified in each dataset
#'   (x column and y column). Specification made with `dplyr::join_by()` are
#'   also accepted.
#'
#' @param radius (float) The cosine distance threshold below which two embeddings
#'   should be considered a match (default is .1).
#'
#' @param exhaustive (boolean) whether to perform an exhaustive
#' nearest-neighbor search to find matching observations. If set to FALSE (the
#' default), HNSW is used and only the 20-closest observations in dataframe a
#' are considered as possible matches to each observation in dataframe b. If
#' set to true, an exahstive match will be perfomed, which takes much more time to run.
#'
#' @param ... additional parameters to be passed to `generate_embeddings`
#'
#' @return A tibble fuzzily-joined on the basis of the variables in `by.` Tries
#'   to adhere to the same standards as the dplyr-joins, and uses the same
#'   logical joining patterns (i.e. inner-join joins and keeps only observations
#'   in both datasets).
#'
#' @rdname neural-joins
#' @export
neural_anti_join <- function(a, b, model, by = NULL, radius = .1, exhaustive = FALSE, ...) {
  neural_join_core(model, a, b,  mode = "anti", by = by, radius, exhaustive, ...)
}

#' @rdname neural-joins
#' @export
neural_inner_join <- function(a, b, model, by = NULL, radius = .1, exhaustive = FALSE, ...) {
  neural_join_core(model, a, b,  mode = "inner", by = by, radius, exhaustive, ...)
}

#' @rdname neural-joins
#' @export
neural_left_join <- function(a, b, model, by = NULL, radius = .1, exhaustive = FALSE, ...) {
  neural_join_core(model, a, b, mode = "left", by = by, radius, exhaustive, ...)
}

#' @rdname neural-joins
#' @export
neural_right_join <- function(a, b, model, by = NULL, radius = .1, exhaustive = FALSE, ...) {
  neural_join_core(model, a, b, mode = "right", by = by, radius, exhaustive, ...)
}

#' @rdname neural-joins
#' @export
neural_full_join <- function(a, b, model, by = NULL, radius = .1, exhaustive = FALSE, ...) {
  neural_join_core(model, a, b,  mode = "full", by = by, radius, exhaustive, ...)
}
