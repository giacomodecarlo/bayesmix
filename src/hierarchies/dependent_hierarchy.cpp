#include "dependent_hierarchy.hpp"

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <stan/math/prim.hpp>

void DependentHierarchy::add_datum(const int id, const Eigen::VectorXd &datum,
                                   const Eigen::VectorXd &covariate) {
  auto it = cluster_data_idx.find(id);
  assert(it == cluster_data_idx.end());
  card += 1;
  log_card = std::log(card);
  update_summary_statistics(datum, covariate, true);
  cluster_data_idx.insert(id);
}

void DependentHierarchy::remove_datum(const int id,
    const Eigen::VectorXd &datum, const Eigen::VectorXd &covariate) {
  update_summary_statistics(datum, covariate, false);
  card -= 1;
  log_card = (card == 0) ? stan::math::NEGATIVE_INFTY : std::log(card);
  auto it = cluster_data_idx.find(id);
  assert(it != cluster_data_idx.end());
  cluster_data_idx.erase(it);
}
