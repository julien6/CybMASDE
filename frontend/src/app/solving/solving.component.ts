import { FormBuilder, Validators, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ChangeDetectionStrategy, Component, OnInit, inject, computed, signal } from '@angular/core';

export interface Algorithm {
  name: string;
  completed: boolean;
  subalgorithms?: Algorithm[];
}

@Component({
  selector: 'app-solving',
  templateUrl: './solving.component.html',
  styleUrls: ['./solving.component.css']
})
export class SolvingComponent implements OnInit {

  private _formBuilder = inject(FormBuilder);

  solvingFormGroup = this._formBuilder.group({
    solvingCtrl: ['', Validators.required],
  });

  editorOptions = { theme: 'vs-dark', language: 'python' };

  envSimModel = "";
  runningOutput = "";
  solvingOutput = "";

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

  private selectedFileName: string | null = null;

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

}
