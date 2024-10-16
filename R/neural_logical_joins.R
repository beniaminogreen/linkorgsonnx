#' @rdname neural-joins
#' @export
neural_anti_join <- function(a, b, by = NULL, radius = .1, exhaustive = FALSE) {
  neural_join_core(a, b, mode = "anti", by = by, radius, exhaustive)
}

#' @rdname neural-joins
#' @export
neural_inner_join <- function(a, b, by = NULL, radius = .1, exhaustive = FALSE) {
  neural_join_core(a, b, mode = "inner", by = by, radius, exhaustive)
}

#' @rdname neural-joins
#' @export
neural_left_join <- function(a, b, by = NULL, radius = .1, exhaustive = FALSE) {
  neural_join_core(a, b, mode = "left", by = by, radius, exhaustive)
}

#' @rdname neural-joins
#' @export
neural_right_join <- function(a, b, by = NULL, radius = .1, exhaustive = FALSE) {
  neural_join_core(a, b, mode = "right", by = by, radius, exhaustive)
}

#' @rdname neural-joins
#' @export
neural_full_join <- function(a, b, by = NULL, radius = .1, exhaustive = FALSE) {
  neural_join_core(a, b, mode = "full", by = by, radius, exhaustive)
}
