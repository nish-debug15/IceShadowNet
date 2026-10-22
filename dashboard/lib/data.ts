import resultsData from '../public/data/results.json';
import curvesData from '../public/data/training_curves.json';
import patchScoresData from '../public/data/per_patch_scores.json';
import metaData from '../public/data/meta.json';
import gradcamManifest from '../public/data/gradcam/manifest.json';
import pipelineManifest from '../public/data/pipeline_stages/manifest.json';

export interface ModelRun {
  model: string;
  variant: string;
  seed: number;
  best_epoch: number;
  val_auc: number;
  val_f1: number;
  tuned_th: number;
  params: number;
  train_time_s: number;
  test_primary: {
    roc_auc: number;
    pr_auc: number;
    f1: number;
    accuracy: number;
    precision: number;
    recall: number;
    support: string;
    prevalence: string;
    f1_default_ci: string;
    f1_conditional_ci: string;
    roc_auc_ci: string;
    pr_auc_ci: string;
  };
  test_clean: {
    roc_auc: number;
    pr_auc: number;
    f1: number;
    accuracy: number;
    precision: number;
    recall: number;
    support: string;
    prevalence: string;
    f1_default_ci: string;
    f1_conditional_ci: string;
    roc_auc_ci: string;
    pr_auc_ci: string;
  };
}

export interface PatchRecord {
  patch_idx: number;
  coords: number[];
  strip_id: number;
  label: number;
  pos_px: number;
  ice_frac: number;
  cpr_mean: number;
  dop_mean: number;
  is_clean: boolean;
  image_path: string;
  scores: Record<string, number>;
}

export interface GradCamSlot {
  slot: string;
  category: string;
  patch_idx: number;
  strip_id: number;
  coords: number[];
  true_label: number;
  trained_model: {
    name: string;
    weights: string;
    score: number;
    threshold: number;
    prediction: number;
    gradcam_image: string;
  };
  untrained_model: {
    name: string;
    weights: string;
    score: number;
    gradcam_image: string;
  };
}

export interface PipelineStage {
  stage: number;
  name: string;
  image: string;
  size_kb: number;
}

export function getResultsData() {
  return resultsData;
}

export function getTrainingCurvesData() {
  return curvesData;
}

export function getPatchScoresData() {
  return patchScoresData;
}

export function getMetaData() {
  return metaData;
}

export function getGradCamData() {
  return gradcamManifest;
}

export function getPipelineStagesData() {
  return pipelineManifest;
}
