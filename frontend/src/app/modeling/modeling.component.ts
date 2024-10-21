import { Component, ChangeDetectionStrategy, OnInit, inject } from '@angular/core';
import { FormBuilder, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { environment } from '../../environments/environment';

@Component({
  selector: 'app-modeling',
  templateUrl: './modeling.component.html',
  styleUrls: ['./modeling.component.css']
})
export class ModelingComponent implements OnInit {

  private _formBuilder = inject(FormBuilder);

  editorOptions = { theme: 'vs-dark', language: 'python' };

  selectedEnvironmentInput = "environmentTraces"
  selectedGoalInput = "goalModel"
  selectedConstraintInput = "constraintModel"

  modelingInput: any = {
    "environmentInput": {
      'environmentApi': { 'fullName': 'Environment API URL', 'content': '' },
      'environmentTraces': { 'fullName': 'Environment Traces', 'content': '' },
      'environmentModel': { 'fullName': 'Observation Transition Function Model', 'content': '' }
    },
    "goalInput": {
      'goalText': { 'fullName': 'Goal Text (alpha)', 'content': '' },
      'goalModel': { 'fullName': 'Reward Function Model', 'content': '' },
      'goalStates': { 'fullName': 'Goal States', 'content': '' }      
    },
    "constraintInput": {
      'constraintText': { 'fullName': 'Constraint Text (alpha)', 'content': '' },
      'constraintModel': { 'fullName': 'Constraint Model', 'content': '' },
    }
  }

  outputEnvironmentModel = ""

  onFileSelected(event: Event, inputCategory: string, inputType: string): void {
    const input = event.target as HTMLInputElement;

    if (input.files && input.files.length > 0) {
      const file: File = input.files[0];
      const reader = new FileReader();
      reader.onload = () => {
          this.modelingInput[inputCategory][inputType]['content'] = reader.result as string;
      };
      reader.readAsText(file);
    }
  }

  openLink(url: string) {
    window.open(url, '_blank');
  }

  constructor() { }

  ngOnInit() {
  }

}
