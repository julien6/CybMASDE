import { Component, OnInit } from '@angular/core';
import { MatDialogRef } from '@angular/material/dialog';
import { FormBuilder, FormControl, FormGroup, Validators } from '@angular/forms';
import { TemplateRef, ViewChild } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';

import 'ace-builds';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/mode-yaml';
import 'ace-builds/src-noconflict/mode-typescript';
import 'ace-builds/src-noconflict/mode-scss';

import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { Inject } from '@angular/core';
import { ServerAPIService } from '../services/serverAPI.service';


@Component({
  selector: 'app-configure-run',
  templateUrl: './configure-run.component.html',
  styleUrls: ['./configure-run.component.css']
})
export class ConfigureRunComponent implements OnInit {

  jsonInputData = '';
  yamlInputData = '';
  appModuleTsData = '';
  scssData = '';
  logsData = '';

  firstFormGroup: FormGroup;

  fullSimulationFormGroup: FormGroup;
  fullEmulationFormGroup: FormGroup;
  select_number_of_iteration_per_episode: FormGroup;

  secondFormGroup: FormGroup;

  selectedMode: string;
  selectedSimulationEngine: string;
  selectedEmulationEngine: string;
  selectedNumIterPerEp: number;
  selectedNumEp: number;
  selectedPauseDuration: number;

  simSettingUpDone: boolean;
  simSettingUpProgressValue: number;

  constructor(public dialog: MatDialog, private serverAPIService: ServerAPIService, @Inject(MAT_DIALOG_DATA) public data: any,
    private _formBuilder: FormBuilder, public dialogRef: MatDialogRef<ConfigureRunComponent>) {
    this.selectedMode = this.data.running_mode;
    this.selectedSimulationEngine = "";
    this.selectedEmulationEngine = "";
    this.selectedNumIterPerEp = 0;
    this.selectedNumEp = 0;
    this.selectedPauseDuration = 0;
    this.simSettingUpDone = false;
    this.simSettingUpProgressValue = 0;
  }

  ngOnInit(): void {

    this.firstFormGroup = this._formBuilder.group({
      select_mode: [this.data.running_mode, Validators.required],
    });

    this.fullSimulationFormGroup = new FormGroup({
      select_simulation_engine: new FormControl(null, Validators.required),
      select_number_of_iteration_per_episode: new FormControl(null, Validators.required),
      select_number_of_episode: new FormControl(null, Validators.required),
      select_pause_duration: new FormControl(0, Validators.required)
    });

    this.fullEmulationFormGroup = new FormGroup({
      select_emulation_engine: new FormControl(null, Validators.required),
    });

  }

  saveSavingPlan() {
    console.log(this.jsonInputData);
  }

  saveDeploymentPlan() {
    console.log(this.yamlInputData);
  }

  onSelectingMode(value: any) {
    this.selectedMode = value;
  }

  onSelectingSimulationEngine(value: any) {
    this.selectedSimulationEngine = value;
  }

  onSelectingNumIterPerEp(value: any) {
    this.selectedNumIterPerEp = value;
  }

  onSelectingEmulationEngine(value: any) {
    this.selectedEmulationEngine = value;
  }

  onSelectingNumEp(value: any) {
    this.selectedNumEp = value;
  }

  onSelectingPauseDuration(value: any) {
    this.selectedPauseDuration = value;
  }

  onLaunchSettingUpSimulation() {

    this.serverAPIService.postRequest("set-simulation?iteration_pause_duration=" + this.selectedPauseDuration + "&simulation_engine=" + this.selectedSimulationEngine +
      "&number_of_iterations=" + this.selectedNumIterPerEp + "&number_of_episodes=" + this.selectedNumEp, {}).subscribe((data: any) => {
        this.logsData = data.logs
        this.simSettingUpDone = true
        this.simSettingUpProgressValue = 100
    })

  }

  openDialogWithRef(ref: TemplateRef<any>, height_percentage: string = "auto", width_percentage: string = "auto") {
    this.dialog.open(ref, {
      height: height_percentage,
      width: width_percentage
    });
  }

}
