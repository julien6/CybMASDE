import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RefiningComponent } from './refining.component';

describe('RefiningComponent', () => {
  let component: RefiningComponent;
  let fixture: ComponentFixture<RefiningComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RefiningComponent]
    })
      .compileComponents();

    fixture = TestBed.createComponent(RefiningComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
