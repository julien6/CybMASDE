import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SolvingComponent } from './solving.component';

describe('SolvingComponent', () => {
  let component: SolvingComponent;
  let fixture: ComponentFixture<SolvingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SolvingComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SolvingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
