import jax.numpy as jnp

def shift_reflect( f, param_dict ):
    #map f(x,y) -> -f(x,-y + 2*pi/8)
    
    #Reflect in real space
    f = jnp.flip(f, axis=-1)
    f = jnp.roll(f, shift=1, axis=-1)

    #Translate in Fourier space
    f = jnp.fft.rfft2(f)
    f = jnp.exp( 1j * param_dict['ky'] * ( 2*jnp.pi)/8 ) * f
    f = jnp.fft.irfft2(f)

    #Reflect the magnetic field
    f = f * jnp.reshape( jnp.array([1,-1]), [2,1,1] )
    return f


if __name__ == "__main__":
    #Checking that the symmetries are implemented correctly to commute with forward time evolution.
    print("Hello")