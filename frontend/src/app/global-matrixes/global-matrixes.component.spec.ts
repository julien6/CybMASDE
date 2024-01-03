import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GlobalMatrixesComponent } from './global-matrixes.component';

describe('GlobalMatrixesComponent', () => {
  let component: GlobalMatrixesComponent;
  let fixture: ComponentFixture<GlobalMatrixesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GlobalMatrixesComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GlobalMatrixesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
