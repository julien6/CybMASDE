import { Component, OnInit, inject } from '@angular/core';
import {FormBuilder, Validators, FormsModule, ReactiveFormsModule} from '@angular/forms';

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

  isLinear = false;

  constructor() { }

  ngOnInit() {
  }

}
