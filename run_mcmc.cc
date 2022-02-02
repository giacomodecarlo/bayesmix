#include <math.h>

#include <fstream>
#include <iostream>

#include "lib/argparse/argparse.h"
#include "src/includes.h"

bool check_args(argparse::ArgumentParser args) {
  if (args["--coll-name"] != std::string("memory")) {
    bayesmix::check_file_is_writeable(args.get<std::string>("--coll-name"));
  }
  if (args["--dens-file"] != std::string("\"\"")) {
    bayesmix::check_file_is_writeable(args.get<std::string>("--dens-file"));
  }
  if (args["--n-cl-file"] != std::string("\"\"")) {
    bayesmix::check_file_is_writeable(args.get<std::string>("--n-cl-file"));
  }
  if (args["--clus-file"] != std::string("\"\"")) {
    bayesmix::check_file_is_writeable(args.get<std::string>("--clus-file"));
  }
  if (args["--best-clus-file"] != std::string("\"\"")) {
    bayesmix::check_file_is_writeable(
        args.get<std::string>("--best-clus-file"));
  }

  return true;
}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser args("bayesmix::run");

  args.add_argument("--algo-params-file")
      .required()
      .help(
          "asciipb file with the parameters of the algorithm, see "
          "the file proto/algorithm_params.proto");

  args.add_argument("--hier-type")
      .required()
      .help(
          "enum string of the hierarchy, see the file "
          "proto/hierarchy_id.proto");

  args.add_argument("--hier-args")
      .required()
      .help(
          "asciipb file with the parameters of the hierarchy, see "
          "the file proto/hierarchy_prior.proto");

  args.add_argument("--mix-type")
      .required()
      .help("enum string of the mixing, see the file proto/mixing_id.proto");

  args.add_argument("--mix-args")
      .required()
      .help(
          "asciipb file with the parameters of the mixing, see "
          "the file proto/mixing_prior.proto");

  args.add_argument("--coll-name")
      .required()
      .default_value("memory")
      .help("If not 'memory', the path where to save the MCMC chains");

  args.add_argument("--data-file")
      .required()
      .help("Path to a .csv file containing the observations (one per row)");

  args.add_argument("--grid-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Path to a csv file containing a grid of points where to "
          "evaluate the (log) predictive density");

  args.add_argument("--dens-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Where to store the output of the (log) predictive "
          "density");

  args.add_argument("--n-cl-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Where to store the MCMC chain of the number of "
          "clusters");

  args.add_argument("--clus-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Where to store the MCMC chain of the cluster "
          "allocations");

  args.add_argument("--best-clus-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Where to store the best cluster allocation found by "
          "minimizing the Binder loss function over the visited partitions");

  args.add_argument("--hier-cov-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Only for dependent models. Path to a csv file with the "
          "covariates used in the hierarchy");

  args.add_argument("--hier-grid-cov-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Only for dependent models and when 'grid-file' is not "
          "empty. "
          "Path to a csv file with the values covariates used in the "
          "hierarchy "
          "on which to evaluate the (log) predictive density");

  args.add_argument("--mix-cov-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Only for dependent models. Path to a csv file with the "
          "covariates used in the mixing");

  args.add_argument("--mix-grid-cov-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Only for dependent models and when 'grid-file' is not "
          "empty. "
          "Path to a csv file with the values covariates used in the mixing "
          "on which to evaluate the (log) predictive density");

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    std::exit(1);
  }

  std::cout << "Running run_mcmc.cc" << std::endl;
  check_args(args);

  // Read algorithm settings proto
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(args.get<std::string>("--algo-params-file"),
                                 &algo_proto);

  // Create factories and objects
  auto &factory_algo = AlgorithmFactory::Instance();
  auto &factory_hier = HierarchyFactory::Instance();
  auto &factory_mixing = MixingFactory::Instance();
  auto algo = factory_algo.create_object(algo_proto.algo_id());
  auto hier = factory_hier.create_object(args.get<std::string>("--hier-type"));
  auto mixing =
      factory_mixing.create_object(args.get<std::string>("--mix-type"));

  BaseCollector *coll;
  if (args["--coll-name"] == std::string("memory")) {
    std::cout << "Creating MemoryCollector" << std::endl;
    coll = new MemoryCollector();
  } else {
    std::cout << "Creating FileCollector, writing to file: "
              << args.get<std::string>("--coll-name") << std::endl;
    coll = new FileCollector(args.get<std::string>("--coll-name"));
  }

  bayesmix::read_proto_from_file(args.get<std::string>("--mix-args"),
                                 mixing->get_mutable_prior());
  bayesmix::read_proto_from_file(args.get<std::string>("--hier-args"),
                                 hier->get_mutable_prior());

  // Read data matrices
  Eigen::MatrixXd data =
      bayesmix::read_eigen_matrix(args.get<std::string>("--data-file"));

  // Set algorithm parameters
  algo->read_params_from_proto(algo_proto);

  // Allocate objects in algorithm
  algo->set_mixing(mixing);
  algo->set_data(data);
  algo->set_hierarchy(hier);

  // Read and set covariates
  if (hier->is_dependent()) {
    Eigen::MatrixXd hier_cov =
        bayesmix::read_eigen_matrix(args.get<std::string>("--hier-cov-file"));
    algo->set_hier_covariates(hier_cov);
  }

  if (mixing->is_dependent()) {
    Eigen::MatrixXd mix_cov =
        bayesmix::read_eigen_matrix(args.get<std::string>("--mix-cov-file"));
    algo->set_mix_covariates(mix_cov);
  }

  // Run algorithm
  algo->run(coll);
  if (args["--grid-file"] != std::string("\"\"") &&
      args["--dens-file"] != std::string("\"\"")) {
    Eigen::MatrixXd grid =
        bayesmix::read_eigen_matrix(args.get<std::string>("--grid-file"));
    Eigen::MatrixXd hier_cov_grid = Eigen::RowVectorXd(0);
    Eigen::MatrixXd mix_cov_grid = Eigen::RowVectorXd(0);
    if (hier->is_dependent()) {
      hier_cov_grid = bayesmix::read_eigen_matrix(
          args.get<std::string>("--hier-grid-cov-file"));
    }
    if (mixing->is_dependent()) {
      mix_cov_grid = bayesmix::read_eigen_matrix(
          args.get<std::string>("--mix-grid-cov-file"));
    }

    std::cout << "Computing log-density..." << std::endl;
    Eigen::MatrixXd dens =
        algo->eval_lpdf(coll, grid, hier_cov_grid, mix_cov_grid);
    bayesmix::write_matrix_to_file(dens, args.get<std::string>("--dens-file"));
    std::cout << "Successfully wrote density to "
              << args.get<std::string>("--dens-file") << std::endl;
  }

  if ((args.get<std::string>("--n-cl-file") != std::string("\"\"")) ||
      (args.get<std::string>("--clus-file") != std::string("\"\"")) ||
      (args.get<std::string>("--best-clus-file") != std::string("\"\""))) {
    Eigen::MatrixXi clusterings(coll->get_size(), data.rows());
    Eigen::VectorXi num_clust(coll->get_size());
    std::vector<Eigen::MatrixXd> mus;
    std::vector<Eigen::MatrixXd> psis;
    std::vector<Eigen::MatrixXd> lambda_row0s;
    std::vector<Eigen::MatrixXd> lambda_lambda_row0s;
    std::vector<Eigen::MatrixXd> lambda_lambda_psis;
    int q = 0;
    int n_iterations = coll->get_size();
    for (int i = 0; i < coll->get_size(); i++) {
      bayesmix::AlgorithmState state;
      coll->get_next_state(&state);

      for (int j = 0; j < data.rows(); j++) {
        clusterings(i, j) = state.cluster_allocs(j);
      }

      for (int j = 0; j < state.cluster_states_size(); j++) {
        if (i == 0) {
          q = bayesmix::to_eigen(state.cluster_states(j).mfa_state().eta())
                  .cols();
          Eigen::MatrixXd mu(coll->get_size(), data.cols());
          Eigen::MatrixXd psi(coll->get_size(), data.cols());
          Eigen::MatrixXd lambda_row0(coll->get_size(), q);
          Eigen::MatrixXd lambda_lambda_row0(coll->get_size(), data.cols());
          Eigen::MatrixXd lambda_lambda_psi(data.cols(), data.cols());
          mu.row(i) =
              bayesmix::to_eigen(state.cluster_states(j).mfa_state().mu());
          psi.row(i) =
              bayesmix::to_eigen(state.cluster_states(j).mfa_state().psi());
          Eigen::MatrixXd lambda =
              bayesmix::to_eigen(state.cluster_states(j).mfa_state().lambda());
          lambda_row0.row(i) = lambda.row(0);
          lambda_lambda_row0.row(i) = (lambda * lambda.transpose()).row(0);
          lambda_lambda_psi = lambda * lambda.transpose() +
                              Eigen::MatrixXd(psi.row(i).asDiagonal());
          lambda_lambda_psis.push_back(lambda_lambda_psi);
          mus.push_back(mu);
          psis.push_back(psi);
          lambda_row0s.push_back(lambda_row0);
          lambda_lambda_row0s.push_back(lambda_lambda_row0);
        } else {
          mus[j].row(i) =
              bayesmix::to_eigen(state.cluster_states(j).mfa_state().mu());
          psis[j].row(i) =
              bayesmix::to_eigen(state.cluster_states(j).mfa_state().psi());
          Eigen::MatrixXd lambda =
              bayesmix::to_eigen(state.cluster_states(j).mfa_state().lambda());
          lambda_row0s[j].row(i) = lambda.row(0);
          lambda_lambda_row0s[j].row(i) = (lambda * lambda.transpose()).row(0);
          lambda_lambda_psis[j] +=
              lambda * lambda.transpose() +
              Eigen::MatrixXd(psis[j].row(i).asDiagonal());
        }
      }
      num_clust(i) = state.cluster_states_size();
    }

    std::string exp =
        std::to_string(data.cols()) + "_" + std::to_string(q) + " cluster n";
    for (int i = 0; i < mus.size(); i++) {
      bayesmix::write_matrix_to_file(mus[i],
                                     exp + std::to_string(i) + " mu.csv");
      bayesmix::write_matrix_to_file(psis[i],
                                     exp + std::to_string(i) + " psi.csv");
      bayesmix::write_matrix_to_file(
          lambda_row0s[i], exp + std::to_string(i) + " lambda_row0s.csv");
      bayesmix::write_matrix_to_file(
          lambda_lambda_row0s[i],
          exp + std::to_string(i) + " lambda__lambda_row0s.csv");
      bayesmix::write_matrix_to_file(
          lambda_lambda_psis[i] / n_iterations,
          exp + std::to_string(i) + " lambda__lambda_psis.csv");
    }

    if (args.get<std::string>("--n-cl-file") != std::string("\"\"")) {
      bayesmix::write_matrix_to_file(num_clust,
                                     args.get<std::string>("--n-cl-file"));
      std::cout << "Successfully wrote number of clusters to "
                << args.get<std::string>("--n-cl-file") << std::endl;
    }

    if (args.get<std::string>("--clus-file") != std::string("\"\"")) {
      bayesmix::write_matrix_to_file(clusterings,
                                     args.get<std::string>("--clus-file"));
      std::cout << "Successfully wrote cluster allocations to "
                << args.get<std::string>("--clus-file") << std::endl;
    }

    if (args.get<std::string>("--best-clus-file") != std::string("\"\"")) {
      Eigen::VectorXi best_clus = bayesmix::cluster_estimate(clusterings);
      bayesmix::write_matrix_to_file(
          best_clus, args.get<std::string>("--best-clus-file"));
      std::cout << "Successfully wrote best cluster allocations to "
                << args.get<std::string>("--best-clus-file") << std::endl;
    }
  }

  std::cout << "End of run_mcmc.cc" << std::endl;
  delete coll;
  return 0;
}
