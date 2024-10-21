import { FormBuilder, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ChangeDetectionStrategy, Component, OnInit, inject, computed, signal } from '@angular/core';

export interface Algorithm {
  name: string;
  completed: boolean;
  subalgorithms?: Algorithm[];
}

@Component({
  selector: 'app-analyzing',
  templateUrl: './analyzing.component.html',
  styleUrls: ['./analyzing.component.css']
})
export class AnalyzingComponent implements OnInit {

  private _formBuilder = inject(FormBuilder);

  analyzingFormGroup = this._formBuilder.group({
    analyzingCtrl: ['', Validators.required],
  });

  analyzingOutput = "";
  editorOptions = { theme: 'vs-dark', language: 'python' };
  runningOutput = "";

  isLinear = false;
  private selectedFileName = '';
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
      this.selectedFileName = file.name;
    }
  }

  constructor() { }

  ngOnInit() {
  }

  readonly algorithm = signal<Algorithm>({
    name: 'All',
    completed: false,
    subalgorithms: [
      {
        name: 'Proximal Policy Optimization', completed: false, subalgorithms: [
          { name: 'MAPPO', completed: false }]
      },
      {
        name: 'Deep Deterministc Policy Gradient', completed: false, subalgorithms: [
          { name: 'MADDPG', completed: false }]
      },
    ],
  });

  kosiaEnabled = false;
  gosiaEnabled = false;

  automatedHPO = false;

  readonly partiallyComplete = computed(() => {
    const algorithm = this.algorithm();
    if (!algorithm.subalgorithms) {
      return false;
    }
    return algorithm.subalgorithms.some(t => t.completed) && !algorithm.subalgorithms.every(t => t.completed);
  });

  update(completed: boolean, index?: number) {
    this.algorithm.update(algorithm => {
      if (index === undefined) {
        algorithm.completed = completed;
        algorithm.subalgorithms?.forEach(t => (t.completed = completed));
      } else {
        algorithm.subalgorithms![index].completed = completed;
        algorithm.completed = algorithm.subalgorithms?.every(t => t.completed) ?? true;
      }
      return { ...algorithm };
    });
  }

}
