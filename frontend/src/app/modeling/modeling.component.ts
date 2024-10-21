import { Component, OnInit, inject } from '@angular/core';
import { FormBuilder, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';

@Component({
  selector: 'app-modeling',
  templateUrl: './modeling.component.html',
  styleUrls: ['./modeling.component.css']
})
export class ModelingComponent implements OnInit {

  private _formBuilder = inject(FormBuilder);

  editorOptions = {theme: 'vs-dark', language: 'python'};

  selectedFileName: string | null = null;
  envSimModel = "";

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
      this.selectedFileName = file.name;
      // Vous pouvez également ajouter votre logique de téléchargement ici.
    }
  }

  selectedEnvironmentInput: string = "";
  selectedGoalInput = "";
  selectedConstraintInput = "";
  
  environmentOptions = [
    { value: 'environmentApi', viewValue: 'Environment API' },
    { value: 'environmentTraces', viewValue: 'Environment Traces' },
    { value: 'environmentModel', viewValue: 'Environment Model' }
  ];

  goalOptions = [
    { value: 'goalText', viewValue: 'Goal Text' },
    { value: 'goalModel', viewValue: 'Goal Model' },
    { value: 'goalStates', viewValue: 'Goal States' }
  ];

  constraintOptions = [
    { value: 'constraintText', viewValue: 'Constraint Text' },
    { value: 'constraintModel', viewValue: 'Constraint Model' },
  ];

  environmentApi = null;
  environmentTraces = '{\n    "trace": {}\n}';
  environmentModel = null;

  goalText = null;
  goalStates = null;
  goalModel = null;

  constraintText = null;
  constraintModel = null;


  // Vous pouvez également créer une méthode pour agir sur la sélection
  onSelectionChange() {
    console.log(this.selectedEnvironmentInput);  // Affiche la valeur sélectionnée dans la console
  }

  modelingFormGroup = this._formBuilder.group({
    modelingCtrl: ['', Validators.required],
  });


  constructor() { }

  ngOnInit() {
  }

}
