import { Component, OnInit, inject } from '@angular/core';
import {FormBuilder, Validators, FormsModule, ReactiveFormsModule} from '@angular/forms';

@Component({
  selector: 'app-transfering',
  templateUrl: './transfering.component.html',
  styleUrls: ['./transfering.component.css']
})
export class TransferingComponent implements OnInit {

  private _formBuilder = inject(FormBuilder);

  transferingFormGroup = this._formBuilder.group({
    transferingCtrl: ['', Validators.required],
  });

  constructor() { }

  private selectedFileName = '';
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
      this.selectedFileName = file.name;
    }
  }

  manualEnabled = false;
  llmEnabled = false;

  transferingOutput = "";

  runningOutput = "";

  editorOptions = { theme: 'vs-dark', language: 'python' };

  ngOnInit() {
  }

}
