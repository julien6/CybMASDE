// =========================
// Root Project Configuration
// =========================

export interface ProjectConfig {
  common: Common;
  modelling: ModellingConfig;
  training: TrainingConfig;
  analyzing: AnalyzingConfig;
  transferring: TransferringConfig;
  refining: RefiningConfig;
}

// =========================
// Common Section
// =========================

export interface Common {
  project_name: string;
  project_description: string;
  label_manager: string;
  project_path: string;
}

// =========================
// Modelling Section
// =========================

export interface ModellingConfig {
  environment_path: string;
  generated_environment: GeneratedEnvironment;
}

export interface GeneratedEnvironment {
  world_model: WorldModel;
  component_functions_path: string;
}

export interface WorldModel {
  jopm: JOPM;
  statistics: Record<string, any>; // Empty or populated object
  used_traces_path: string;
}

export interface JOPM {
  autoencoder: AutoencoderModel;
  rdlm: RdlmModel;
  initial_joint_observations: string;
}

export interface AutoencoderModel {
  statistics: Record<string, any>;
  model: string;
  hyperparameters: AutoencoderHyperparameters;
  max_mean_square_error: string | number;
}

export interface RdlmModel {
  statistics: Record<string, any>;
  model: string;
  hyperparameters: RdlmHyperparameters;
  max_mean_square_error: string | number;
}

// --- Hyperparameters Models ---

export interface AutoencoderHyperparameters {
  latent_dim: number[];
  hidden_dim: number[];
  n_layers: number[];
  activation: string[];
  lr: number[];
  batch_size: number[];
  kl_weight: number[];
}

export interface RdlmHyperparameters {
  rchs_dim: number[];
  rnn_type: string[];
  rnn_hidden_dim: number[];
  rnn_layers: number[];
  rnn_activation: string[];
  learning_rate: number[];
  mlp_layers: number[];
  mlp_hidden_dim: number[];
  mlp_activation: string[];
}

// =========================
// Training Section
// =========================

export interface TrainingConfig {
  hyperparameters: TrainingHyperparameters;
  organizational_specifications: string;
  joint_policy: string;
  statistics: Record<string, any>;
}

export interface TrainingHyperparameters {
  algorithms: TrainingAlgorithms;
  mean_threshold: number;
  max_timesteps_total: number;
  std_threshold: number;
  window_size: number;
  num_gpus: number;
  num_workers: number;
  checkpoint_freq: number;
}

export interface TrainingAlgorithms {
  mappo: {
    algorithm: MappoAlgorithm;
    model: MappoModel;
  };
}

export interface MappoAlgorithm {
  batch_mode: string;
  lr: number;
  entropy_coeff: number;
  num_sgd_iter: number;
  clip_param: number;
  use_gae: boolean;
  lambda: number;
  vf_loss_coeff: number;
  kl_coeff: number;
  vf_clip_param: number;
}

export interface MappoModel {
  core_arch: string;
  encode_layer: string;
}

// =========================
// Analyzing Section
// =========================

export interface AnalyzingConfig {
  hyperparameters: Record<string, any>;
  statistics: Record<string, any>;
  figures_path: string;
  post_training_trajectories_path: string;
  inferred_organizational_specifications: string;
}

// =========================
// Transferring Section
// =========================

export interface TransferringConfig {
  configuration: TransferringConfiguration;
}

export interface TransferringConfiguration {
  trajectory_retrieve_frequency: number;
  trajectory_batch_size: number;
  deploy_mode: 'REMOTE' | 'DIRECT';
  environment_api: string;
  max_nb_iteration: number;
}

// =========================
// Refining Section
// =========================

export interface RefiningConfig {
  max_refinement_cycles: number;
  auto_continue_refinement: boolean;
}
